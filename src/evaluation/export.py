"""Safe, self-contained exports for persisted model comparisons."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from dataclasses import dataclass
from typing import Any, Literal

ExportFormat = Literal["json", "csv", "html"]


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    content: str
    media_type: str
    filename: str
    sha256: str


def export_comparison(comparison: dict[str, Any], format: ExportFormat) -> ExportArtifact:
    """Serialize comparison evidence without reinterpreting any metric values."""
    comparison_id = str(comparison["comparison_id"])
    if format == "json":
        content = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True)
        return _artifact(content, "application/json", f"{comparison_id}.json")
    if format == "csv":
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=("metric", "value", "unit"))
        writer.writeheader()
        for name, metric in sorted(comparison.get("metric_deltas", {}).items()):
            writer.writerow({"metric": name, "value": metric.get("value"), "unit": metric.get("unit", "")})
        recovery = comparison.get("recovery_report", {}).get("recovery_rate")
        if recovery:
            writer.writerow({"metric": "recovery_rate", "value": recovery.get("percent_value"), "unit": recovery.get("unit", "")})
        return _artifact(output.getvalue(), "text/csv", f"{comparison_id}.csv")
    if format == "html":
        rows = "".join(
            f"<tr><th>{html.escape(str(name))}</th><td>{html.escape(str(metric.get('value')))}</td><td>{html.escape(str(metric.get('unit', '')))}</td></tr>"
            for name, metric in sorted(comparison.get("metric_deltas", {}).items())
        )
        content = (
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>AdverTest comparison</title></head>"
            "<body><h1>AdverTest comparison</h1>"
            f"<p>Comparison: {html.escape(comparison_id)}</p>"
            f"<p>Paired: {html.escape(str(bool(comparison.get('paired'))))}</p>"
            "<table><thead><tr><th>Metric</th><th>Value</th><th>Unit</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></body></html>"
        )
        return _artifact(content, "text/html", f"{comparison_id}.html")
    raise ValueError(f"unsupported comparison export format: {format}")


def _artifact(content: str, media_type: str, filename: str) -> ExportArtifact:
    return ExportArtifact(
        content=content,
        media_type=media_type,
        filename=filename,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
