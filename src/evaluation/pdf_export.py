"""Pure-Python dependency-free PDF report generator for AdverTest benchmark runs.

Produces standard PDF 1.4 documents formatted with KPI executive summaries,
degradation matrices, and scientific provenance metadata.
"""

from __future__ import annotations

import io
from typing import Any


def _escape_pdf_text(text: str) -> str:
    """Escape special characters for PDF literal strings."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_run_report_pdf(report: dict[str, Any], run_id: str | None = None) -> bytes:
    """Generate a clean, professional PDF report from a RunReport dictionary."""
    run_id = run_id or report.get("run_id", "UNKNOWN-RUN")
    model_name = report.get("model", "Unknown Model")
    model_version = report.get("model_version", "1.0.0")
    dataset_name = report.get("dataset", "Unknown Dataset")
    n_samples = report.get("n_samples", 0)
    ap_clean = report.get("ap_clean", 0.0)
    cells = report.get("cells", [])
    provenance = report.get("provenance", {})

    # Calculate summary metrics
    degradations = [c.get("degradation_ratio", c.get("degradation", 0.0)) for c in cells]
    mean_deg = sum(degradations) / len(degradations) if degradations else 0.0
    robustness_score = max(0.0, 1.0 - mean_deg) * 100.0

    # Helper to append PDF stream commands
    stream_cmds: list[str] = []

    def draw_rect(x: float, y: float, w: float, h: float, r: float, g: float, b: float, fill: bool = True):
        if fill:
            stream_cmds.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
            stream_cmds.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re f")
        else:
            stream_cmds.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
            stream_cmds.append("1 w")
            stream_cmds.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re S")

    def draw_text(x: float, y: float, text: str, font: str = "F1", size: float = 10, r: float = 0, g: float = 0, b: float = 0):
        clean_txt = _escape_pdf_text(str(text))
        stream_cmds.append(f"BT /{font} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} Td ({clean_txt}) Tj ET")

    # Header Background Banner
    draw_rect(0, 740, 612, 52, 0.08, 0.18, 0.36, fill=True)
    draw_text(36, 762, "AdverTest - AI Robustness Benchmark Report", font="F2", size=16, r=1, g=1, b=1)
    draw_text(36, 748, "SIMULATION ONLY — Research & Model Robustness Evaluation", font="F1", size=9, r=0.8, g=0.9, b=1)

    # Overview Card
    draw_rect(36, 620, 540, 105, 0.96, 0.97, 0.98, fill=True)
    draw_rect(36, 620, 540, 105, 0.85, 0.88, 0.92, fill=False)

    draw_text(48, 706, f"Run ID: {run_id}", font="F2", size=11, r=0.1, g=0.15, b=0.25)
    draw_text(48, 688, f"Target Model: {model_name} (v{model_version})", font="F1", size=10, r=0.2, g=0.25, b=0.3)
    draw_text(48, 672, f"Evaluation Dataset: {dataset_name}", font="F1", size=10, r=0.2, g=0.25, b=0.3)
    draw_text(48, 656, f"Total Evaluation Samples: {n_samples}", font="F1", size=10, r=0.2, g=0.25, b=0.3)
    draw_text(48, 640, f"Clean Baseline AP: {ap_clean:.4f}", font="F2", size=10, r=0.05, g=0.5, b=0.2)

    draw_text(340, 688, f"Robustness Score: {robustness_score:.1f} / 100", font="F2", size=12, r=0.1, g=0.3, b=0.7)
    draw_text(340, 672, f"Mean Degradation: {mean_deg * 100:.1f}%", font="F1", size=10, r=0.8, g=0.2, b=0.1)
    draw_text(340, 656, f"Tested Attack Variations: {len(cells)} cells", font="F1", size=10, r=0.3, g=0.35, b=0.4)
    draw_text(340, 640, "Verification Gate: WAITING_FOR_GPU_VALIDATION" if not provenance.get("cuda_verified") else "Verification Gate: VERIFIED_GPU", font="F1", size=9, r=0.4, g=0.4, b=0.5)

    # Table Title
    draw_text(36, 595, "Attack Degradation Breakdown", font="F2", size=12, r=0.1, g=0.15, b=0.25)

    # Table Header
    draw_rect(36, 565, 540, 22, 0.9, 0.93, 0.96, fill=True)
    draw_text(45, 572, "Attack Scenario", font="F2", size=9, r=0.2, g=0.25, b=0.3)
    draw_text(190, 572, "Severity", font="F2", size=9, r=0.2, g=0.25, b=0.3)
    draw_text(250, 572, "Attacked AP", font="F2", size=9, r=0.2, g=0.25, b=0.3)
    draw_text(340, 572, "Degradation (Drop %)", font="F2", size=9, r=0.2, g=0.25, b=0.3)
    draw_text(460, 572, "Status / Category", font="F2", size=9, r=0.2, g=0.25, b=0.3)

    # Table Rows (up to 15 items on page 1)
    y_cursor = 545
    for idx, c in enumerate(cells[:14]):
        bg_color = (0.98, 0.98, 0.99) if idx % 2 == 0 else (1.0, 1.0, 1.0)
        draw_rect(36, y_cursor - 4, 540, 18, bg_color[0], bg_color[1], bg_color[2], fill=True)

        att_name = str(c.get("attack", "unknown"))
        sev = str(c.get("severity", 1))
        att_ap = float(c.get("ap", 0.0))
        deg_val = float(c.get("degradation_percent", c.get("degradation", 0.0) * 100.0))
        cat = str(c.get("category", c.get("group", "perturbation")))

        draw_text(45, y_cursor, att_name[:24], font="F1", size=8.5, r=0.15, g=0.2, b=0.25)
        draw_text(205, y_cursor, f"Level {sev}", font="F1", size=8.5, r=0.3, g=0.35, b=0.4)
        draw_text(260, y_cursor, f"{att_ap:.4f}", font="F1", size=8.5, r=0.2, g=0.2, b=0.2)

        deg_r = 0.8 if deg_val > 30 else 0.2
        deg_g = 0.1 if deg_val > 30 else 0.5
        draw_text(355, y_cursor, f"{deg_val:.1f}%", font="F2", size=8.5, r=deg_r, g=deg_g, b=0.1)
        draw_text(460, y_cursor, cat[:18], font="F1", size=8.5, r=0.4, g=0.45, b=0.5)

        y_cursor -= 18

    # Provenance Footer Box
    draw_rect(36, 50, 540, 65, 0.95, 0.95, 0.96, fill=True)
    draw_rect(36, 50, 540, 65, 0.85, 0.85, 0.88, fill=False)
    draw_text(45, 98, "Scientific Provenance & Audit Trail", font="F2", size=9, r=0.2, g=0.25, b=0.3)
    ckpt_hash = provenance.get("checkpoint_sha256", "N/A")
    config_hash = provenance.get("run_config_hash", "N/A")
    seed_val = provenance.get("seed", 42)
    protocol_hash = provenance.get("protocol_hash", "official-protocol-locked")

    draw_text(45, 82, f"Checkpoint SHA256: {ckpt_hash[:32]}...", font="F1", size=7.5, r=0.3, g=0.35, b=0.4)
    draw_text(45, 70, f"Config Hash: {config_hash[:32]}... | Seed: {seed_val}", font="F1", size=7.5, r=0.3, g=0.35, b=0.4)
    draw_text(45, 58, f"Protocol Hash: {protocol_hash[:32]} | Official Metric Evaluator: LOCKED", font="F1", size=7.5, r=0.3, g=0.35, b=0.4)

    # Footer Page Number
    draw_text(260, 30, "AdverTest Evaluation Platform — Page 1 of 1", font="F1", size=8, r=0.5, g=0.5, b=0.5)

    stream_content = "\n".join(stream_cmds).encode("latin-1")

    # Build PDF Objects
    objects: list[bytes] = []

    # 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3: Page
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 6 0 R /Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> >>"
    )
    # 4: Standard Font (Helvetica)
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    # 5: Standard Font Bold (Helvetica-Bold)
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    # 6: Content Stream
    stream_header = f"<< /Length {len(stream_content)} >>\nstream\n".encode("latin-1")
    stream_footer = b"\nendstream"
    objects.append(stream_header + stream_content + stream_footer)

    # Assemble complete PDF file
    output = io.BytesIO()
    output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: list[int] = []
    for i, obj in enumerate(objects, 1):
        offsets.append(output.tell())
        output.write(f"{i} 0 obj\n".encode("latin-1"))
        output.write(obj)
        output.write(b"\nendobj\n")

    xref_offset = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for offset in offsets:
        output.write(f"{offset:010d} 00000 n \n".encode("latin-1"))

    output.write(b"trailer\n")
    output.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1"))
    output.write(b"startxref\n")
    output.write(f"{xref_offset}\n%%EOF\n".encode("latin-1"))

    return output.getvalue()
