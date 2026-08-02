import os

CORRUPTIONS = [
    "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    "snow", "frost", "fog", "brightness", "spatter",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate",
]

BASE_CODE = '''"""Shared logic for ImageNet-C corruptions."""

from __future__ import annotations

import random
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from src.attacks.base import AttackContext, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None


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
        random.seed(seed)
        np.random.seed(seed)

        orig_rng = np.random.default_rng
        with patch('numpy.random.default_rng', lambda *a, **k: orig_rng(seed)):
            corrupted_uint8 = corrupt(img_uint8, corruption_name=self.name, severity=severity)

        new_image = corrupted_uint8.astype(np.float32) / 255.0
        return sample.with_image(new_image)
'''

TEMPLATE_CODE = '''"""Group A: {title}."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class {class_name}(ImageCorruptionBase):
    """{title} from ImageNet-C."""

    name: ClassVar[str] = "{name}"
'''

GAUSSIAN_NOISE_CODE = '''"""Group A: Additive Gaussian sensor noise."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class GaussianNoiseParams(AttackParams):
    sigma_per_severity: tuple[float, ...] = (0.04, 0.06, 0.09, 0.13, 0.18)


@ATTACKS.register
class GaussianNoise(BaseAttack):
    """Additive Gaussian noise, ImageNet-C severity ladder."""

    name: ClassVar[str] = "gaussian_noise"
    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "core"
    reference: ClassVar[str] = "Hendrycks & Dietterich, ICLR 2019 (arXiv:1903.12261)"
    params_model: ClassVar[type[AttackParams]] = GaussianNoiseParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        sigma = self.level(severity, self.params.sigma_per_severity)
        noise = ctx.rng.normal(0, sigma, sample.image.shape)
        return sample.with_image(sample.image + noise.astype(np.float32))
'''

dir_path = "src/attacks/corruption"
os.makedirs(dir_path, exist_ok=True)

with open(os.path.join(dir_path, "_base.py"), "w") as f:
    f.write(BASE_CODE)

with open(os.path.join(dir_path, "gaussian_noise.py"), "w") as f:
    f.write(GAUSSIAN_NOISE_CODE)

for name in CORRUPTIONS:
    class_name = "".join(word.capitalize() for word in name.split("_"))
    title = name.replace("_", " ").capitalize()
    content = TEMPLATE_CODE.format(title=title, class_name=class_name, name=name)
    with open(os.path.join(dir_path, f"{name}.py"), "w") as f:
        f.write(content)

if os.path.exists(os.path.join(dir_path, "common.py")):
    os.remove(os.path.join(dir_path, "common.py"))
