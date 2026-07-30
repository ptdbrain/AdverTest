"""Group E — adversarial patches, physically plausible (plan §2).

All slots open. A patch attack optimises the patch pixels instead of the whole
image, so it needs several gradient steps per sample (``cost_class =
"expensive"``) and, per plan §2, EOT augmentation plus the TV + NPS loss terms.
Two shapes of contribution fit the same plugin contract:

* per-image patch — optimise inside ``apply`` using ``ctx.model``;
* universal patch — train once (script under ``scripts/``), then have ``apply``
  only paste the stored patch, keeping the run cheap and reproducible.

References: Brown et al. 2017, Liu et al. (DPatch) 2018, Thys et al. 2019,
Athalye et al. (EOT) 2018.
"""
