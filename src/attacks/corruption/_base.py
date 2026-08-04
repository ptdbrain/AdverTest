"""Shared logic for ImageNet-C corruptions."""

from __future__ import annotations

import random
from threading import Lock
from typing import ClassVar

import numpy as np

from src.attacks.base import AttackContext, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample

try:
    # imagecorruptions 1.1.2 still passes the removed ``multichannel``
    # argument to modern scikit-image. Keep the published attack formulas and
    # adapt only that compatibility boundary.
    import inspect

    # imagecorruptions 1.1.2 still references the NumPy 1.x alias removed in
    # NumPy 2.0. Add the compatibility alias locally before importing its
    # corruption implementations.
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]

    from imagecorruptions import corrupt
    from imagecorruptions import corruptions as _imagecorruptions

    if "multichannel" not in inspect.signature(_imagecorruptions.gaussian).parameters:
        _skimage_gaussian = _imagecorruptions.gaussian

        def _gaussian_compat(
            image: np.ndarray,
            sigma: float = 1.0,
            multichannel: bool = False,
            **kwargs: object,
        ) -> np.ndarray:
            if multichannel:
                kwargs["channel_axis"] = -1
            return _skimage_gaussian(image, sigma=sigma, **kwargs)

        _imagecorruptions.gaussian = _gaussian_compat
except ImportError:
    corrupt = None


_RNG_LOCK = Lock()


class ImageCorruptionBase(BaseAttack):
    """Base class for all ImageNet-C corruptions."""

    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "core"
    reference: ClassVar[str] = "Hendrycks & Dietterich, ICLR 2019 (imagecorruptions)"

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        if corrupt is None:
            raise RuntimeError("Please install 'imagecorruptions' to use this attack")
        img_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)

        seed = int(ctx.rng.integers(0, 2**31 - 1))
        # imagecorruptions predates Generator and reads module-level RNG state.
        # Isolate that legacy API and restore both RNGs before returning.
        with _RNG_LOCK:
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            random_noise = getattr(
                getattr(_imagecorruptions, "sk", None), "util", None
            )
            original_random_noise = (
                getattr(random_noise, "random_noise", None) if random_noise else None
            )

            def _seeded_random_noise(image: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
                kwargs.setdefault("rng", np.random.default_rng(seed))
                return original_random_noise(image, *args, **kwargs)

            try:
                random.seed(seed)
                np.random.seed(seed)
                if original_random_noise is not None:
                    random_noise.random_noise = _seeded_random_noise
                corrupted_uint8 = corrupt(
                    img_uint8,
                    corruption_name=self.name,
                    severity=severity,
                )
            finally:
                if original_random_noise is not None:
                    random_noise.random_noise = original_random_noise
                random.setstate(python_state)
                np.random.set_state(numpy_state)

        new_image = corrupted_uint8.astype(np.float32) / 255.0
        return sample.with_image(new_image)
