"""Group D — white-box adversarial attacks (plan §2).

These are the only attacks that touch the model: they set ``needs_model`` and
``needs_gradients``, receive the adapter through ``ctx.model``, and read
gradients as plain numpy via ``ctx.model.input_gradient(sample)``. That bridge
is deliberate — no attack plugin imports torch, so the catalog stays importable
in CI and the same attack code works for any framework behind an adapter.

``fgsm.py`` is the worked example. Open slots: PGD, MI-FGSM, C&W, TOG
(vanishing / fabrication / mislabeling), DAG, PGD-for-SAM2.
"""
