"""Low-memory SAM2.1 image-prompt runtime for a dedicated CPU worker.

The regular API process keeps the full-resolution float32 adapter contract.
Render Free cannot hold that model alongside the API, so this worker builds the
same checkpoint in float16 and embeds at 128px before restoring the official
decoder feature sizes.  It remains real SAM2 inference; the reduced-resolution
mode is recorded by the worker and is not a scientific benchmark claim.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np


class _HalfTransforms:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __call__(self, image: Any) -> Any:
        return self.inner(image).half()

    def transform_coords(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.transform_coords(*args, **kwargs)

    def transform_boxes(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.transform_boxes(*args, **kwargs)

    def postprocess_masks(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.postprocess_masks(*args, **kwargs)


def build_predictor(weights: str, config: str, *, device: str = "cpu") -> Any:
    """Build an official SAM2.1 predictor with a bounded CPU footprint."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.utils.transforms import SAM2Transforms

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = build_sam2(config, weights, device=device)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval()

    # Hiera Small's optional skip tensors are useful for quality but add a
    # large pair of 256x256/128x128 activations. Preserve the official decoder
    # layers while using the same path with zero skip contribution.
    decoder = model.sam_mask_decoder
    if decoder.use_high_res_features:
        decoder.output_upscaling = torch.nn.Sequential(*list(decoder.output_upscaling))
        decoder.use_high_res_features = False

    # Official SAM2's positional encoding unconditionally casts coordinates to
    # float32. Keep its tiny random matrix in float32, then return half output
    # so the surrounding half model does not promote the decoder.
    position_encoding = model.sam_prompt_encoder.pe_layer

    def _pe_encoding(self: Any, coords: Any) -> Any:
        coords = (2 * coords.float() - 1) @ self.positional_encoding_gaussian_matrix.float()
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1).to(next(model.parameters()).dtype)

    position_encoding._pe_encoding = types.MethodType(_pe_encoding, position_encoding)

    prompt_encoder = model.sam_prompt_encoder
    original_forward = prompt_encoder.forward

    def _prompt_forward(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        sparse, dense = original_forward(*args, **kwargs)
        dtype = next(model.parameters()).dtype
        return sparse.to(dtype), dense.to(dtype)

    prompt_encoder.forward = _prompt_forward

    class _HalfPredictor(SAM2ImagePredictor):
        def _prep_prompts(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
            prepared = super()._prep_prompts(*args, **kwargs)
            dtype = next(self.model.parameters()).dtype
            return tuple(
                value.to(dtype) if value is not None and getattr(value, "is_floating_point", lambda: False)() else value
                for value in prepared
            )  # type: ignore[return-value]

    predictor = _HalfPredictor(model)
    predictor._transforms = _HalfTransforms(SAM2Transforms(resolution=128, mask_threshold=0.0))
    predictor._bb_feat_sizes = [(32, 32), (16, 16), (8, 8)]
    return predictor


def prepare_features(predictor: Any) -> None:
    """Restore decoder feature sizes after the 128px backbone pass."""
    import torch.nn.functional as functional

    features = predictor._features
    features["image_embed"] = functional.interpolate(
        features["image_embed"].half(), size=(64, 64), mode="bilinear", align_corners=False
    ).half()
    features["high_res_feats"] = []


def predict_masks(predictor: Any, image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[tuple[np.ndarray, float]]:
    """Run one real image embedding plus bounded batched official mask decodes."""
    predictor.set_image(np.ascontiguousarray(image, dtype=np.uint8))
    prepare_features(predictor)
    if not boxes:
        return []
    results: list[tuple[np.ndarray, float]] = []
    for start in range(0, len(boxes), 4):
        box_batch = boxes[start : start + 4]
        masks, scores, _ = predictor.predict(box=np.asarray(box_batch, dtype=np.float32), multimask_output=False)
        if len(masks) != len(box_batch) or len(scores) != len(box_batch):
            raise RuntimeError("SAM2 predictor returned an unexpected number of masks")
        for mask, score in zip(masks, scores, strict=True):
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                mask_array = mask_array[0]
            results.append((mask_array, float(np.asarray(score).reshape(-1)[0])))
    return results


def unload_predictor(predictor: Any | None) -> None:
    """Release the dedicated worker's current model before switching weights."""
    if predictor is None:
        return
    predictor.reset_predictor()
    predictor.model = None
    import gc

    gc.collect()
    try:
        from src.core.memory import trim_memory

        trim_memory()
    except ImportError:
        return
