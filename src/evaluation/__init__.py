"""Evaluation layer.

Implemented here (the minimum needed to close the loop attack -> number):

* :func:`src.evaluation.detection_metrics.iou` and greedy matching,
* :func:`src.evaluation.detection_metrics.average_precision` — AP at a single IoU
  threshold, macro-averaged over classes (AP50 by default),
* :class:`src.evaluation.report.RunReport` — clean AP, per-cell AP, and
  degradation ``D(c, s)``.

Open slots from plan §3, one function each — add them to a new module in this
package and extend :class:`~src.evaluation.report.RunReport`:

* COCO AP@[.50:.95] via ``pycocotools`` (replacing the single-threshold AP),
* ``mPC`` / ``rPC`` (Michaelis et al.), ``RR(c)``, ``mCE`` (Robo3D),
* ``RA(s)`` robustness-accuracy curve per severity,
* ASR at object / image / targeted level,
* segmentation (mIoU, Boundary IoU) and 3D (mAP, NDS, AP3D) metrics,
* bootstrap 95 % confidence intervals,
* ``RobustScore`` 0–100 with per-category weights.
"""
