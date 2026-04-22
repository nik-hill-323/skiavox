"""Clinical report and PDF export for Skiavox."""

from __future__ import annotations

import uuid
from datetime import datetime
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Tuple

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _split_findings(
    predictions: Dict[str, float], urgency_map: Dict[str, str], thresholds: Dict[str, float]
) -> Dict[str, list[Tuple[str, float]]]:
    grouped = {"CRITICAL": [], "HIGH": [], "MODERATE": [], "LOW": [], "NORMAL": []}
    for pathology, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        thr = thresholds.get(pathology, 0.50)
        urgency = urgency_map.get(pathology, "LOW")
        if prob >= thr:
            grouped[urgency].append((pathology, prob))
        else:
            grouped["NORMAL"].append((pathology, prob))
    return grouped


def _recommendations(grouped: Dict[str, list[Tuple[str, float]]]) -> list[str]:
    recs = []
    names = {name for bucket in grouped.values() for name, _ in bucket}
    if any(n == "Pneumothorax" for n, _ in grouped["CRITICAL"]):
        recs.append("URGENT: Contact on-call physician immediately.")
    if "Pneumonia" in names:
        recs.append("Recommend: CBC, CRP, consider antibiotic therapy.")
    if "Mass" in names:
        recs.append("Recommend: CT chest with contrast within 48 hours.")
    if "Cardiomegaly" in names:
        recs.append("Recommend: Echocardiogram and cardiology referral.")
    if not recs:
        recs.append("No acute findings. Routine follow-up as clinically indicated.")
    return recs


def generate_clinical_report(
    predictions: Dict[str, float],
    confidence_intervals: Dict[str, float],
    agreement_score: float,
    patient_info: Dict[str, Any],
    pathology_descriptions: Dict[str, str],
    urgency_map: Dict[str, str],
    clinical_thresholds: Dict[str, float],
) -> str:
    """Generate detailed FDA-style textual clinical report."""
    grouped = _split_findings(predictions, urgency_map, clinical_thresholds)
    recs = _recommendations(grouped)
    analysis_id = str(uuid.uuid4())
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quality = patient_info.get("quality", {})

    executive = "No acute findings detected above pathology-specific thresholds."
    if grouped["CRITICAL"]:
        executive = (
            f"Critical finding identified: {grouped['CRITICAL'][0][0]}. "
            "Immediate specialist review is advised."
        )
    elif grouped["HIGH"]:
        executive = (
            f"High-priority findings include {', '.join(p for p, _ in grouped['HIGH'][:2])}. "
            "Prompt clinical correlation is recommended."
        )

    lines = [
        "=================================================================",
        "              SKIAVOX - CLINICAL ANALYSIS REPORT",
        "                  RESEARCH PROTOTYPE v2.0",
        "=================================================================",
        "PATIENT INFORMATION:",
        f"  Study Date    : {date_str}",
        f"  Analysis ID   : {analysis_id}",
        "  AI Model      : Ensemble (DenseNet121-NIH + DenseNet121-CheX + ResNet50)",
        f"  Model Agreement: {agreement_score:.1f}% consensus across 3 independent models",
        f"  Patient ID    : {patient_info.get('patient_id', 'Unknown')}",
        f"  Patient Age   : {patient_info.get('age', 'Unknown')}",
        f"  Patient Sex   : {patient_info.get('sex', 'Unknown')}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "EXECUTIVE SUMMARY:",
        executive,
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "CLINICAL FINDINGS BY PRIORITY:",
        "",
        "🔴 CRITICAL - IMMEDIATE ACTION REQUIRED:",
    ]
    lines.extend(
        [f" - {n} ({p*100:.1f}%) → {pathology_descriptions.get(n, '')}" for n, p in grouped["CRITICAL"]]
        or [" - None"]
    )
    lines.append("")
    lines.append("🟠 HIGH PRIORITY:")
    lines.extend(
        [f" - {n} ({p*100:.1f}%) → {pathology_descriptions.get(n, '')}" for n, p in grouped["HIGH"]]
        or [" - None"]
    )
    lines.append("")
    lines.append("🟡 MODERATE - FOLLOW UP RECOMMENDED:")
    lines.extend(
        [f" - {n} ({p*100:.1f}%) → {pathology_descriptions.get(n, '')}" for n, p in grouped["MODERATE"]]
        or [" - None"]
    )
    lines.append("")
    lines.append("🟢 LOW - MONITOR:")
    lines.extend(
        [f" - {n} ({p*100:.1f}%) → {pathology_descriptions.get(n, '')}" for n, p in grouped["LOW"]]
        or [" - None"]
    )
    lines.append("")
    lines.append("✅ WITHIN NORMAL LIMITS:")
    lines.extend([f" - {n} ({p*100:.1f}%)" for n, p in grouped["NORMAL"][:8]] or [" - None"])

    lines.extend(
        [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "MODEL CONFIDENCE ANALYSIS:",
            "Pathology                 | Probability | CI (std) | Threshold",
            "--------------------------|-------------|----------|----------",
        ]
    )
    for pathology, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        ci = confidence_intervals.get(pathology, 0.0)
        thr = clinical_thresholds.get(pathology, 0.50)
        lines.append(f"{pathology:<25} | {prob*100:>9.2f}% | {ci:>7.3f} | {thr:>8.2f}")

    lines.extend(
        [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "CLINICAL RECOMMENDATIONS:",
        ]
    )
    lines.extend([f" - {r}" for r in recs])
    lines.extend(
        [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "TECHNICAL QUALITY ASSESSMENT:",
            f"Image Quality Score: {quality.get('quality_score', 'N/A')} ({quality.get('quality_label', 'N/A')})",
            "Positioning: Standard PA view assumed",
            f"Penetration: {'adequate' if quality.get('brightness', 'normal') == 'normal' else 'potentially suboptimal'}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "⚠️  DISCLAIMER: Research prototype only. Not FDA cleared.",
            "    Not for clinical diagnosis. Review by licensed radiologist required.",
            "=================================================================",
        ]
    )
    return "\n".join(lines)


def generate_pdf_report(report_text: str, heatmap_image: Image.Image) -> bytes:
    """Create formatted report PDF and return bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=24, rightMargin=24)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Skiavox - Clinical Analysis Report", styles["Title"]))
    story.append(Spacer(1, 8))

    with NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        heatmap_image.save(tmp.name, format="PNG")
        story.append(RLImage(tmp.name, width=360, height=240))
        story.append(Spacer(1, 10))

        table_data = [["Section", "Details"], ["Prototype", "Research use only"], ["FDA Status", "Not cleared"]]
        t = Table(table_data, colWidths=[120, 360])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))

        for line in report_text.splitlines():
            if line.strip():
                story.append(Paragraph(line.replace(" ", "&nbsp;"), styles["Code"]))

        doc.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
