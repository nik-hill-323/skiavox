"""Skiavox – Advanced Diagnostic Imaging Platform."""

from __future__ import annotations

import importlib.util
import tempfile
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from dicom_handler import estimate_image_quality, is_dicom, load_dicom
from gradcam_utils import generate_comparison_grid, generate_eigencam, generate_gradcam_plus_plus
from patient_history import PatientHistory
from report_generator import generate_clinical_report, generate_pdf_report
from risk_scorer import RiskScorer
from second_opinion import SecondOpinionEngine


# ---------------------------------------------------------------------------
# Bootstrap – load model.py via importlib to avoid top-level `model` namespace
# collision with torchxrayvision internals.
# ---------------------------------------------------------------------------

def _load_predictor_class():
    model_path = Path(__file__).with_name("model.py")
    spec = importlib.util.spec_from_file_location("skiavox_local_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load predictor from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnsembleChestPredictor


EnsembleChestPredictor = _load_predictor_class()

PREDICTOR: Optional[EnsembleChestPredictor] = None
HISTORY = PatientHistory()
SECOND_OPINION_ENGINE = SecondOpinionEngine()
LAST_ANALYSIS: Dict[str, Any] = {}


def get_predictor() -> EnsembleChestPredictor:
    """Return or initialise the globally cached ensemble predictor."""
    global PREDICTOR
    if PREDICTOR is None:
        print("[Skiavox] Initialising ensemble models...")
        try:
            PREDICTOR = EnsembleChestPredictor()
            print("[Skiavox] All models ready.")
        except Exception as exc:
            print(f"[Skiavox] Model init failed: {exc}")
            raise RuntimeError(
                "Model loading failed. Check torch/torchxrayvision installation."
            ) from exc
    return PREDICTOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RISK_COLOR = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MODERATE": "#eab308",
    "LOW":      "#22c55e",
}


def _risk_label(level: str) -> str:
    return {
        "CRITICAL": "CRITICAL",
        "HIGH":     "HIGH",
        "MODERATE": "MODERATE",
        "LOW":      "LOW",
    }.get(level, level)


def _build_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart, colour-coded by urgency, with threshold line."""
    colors = df["Risk"].map(_RISK_COLOR).fillna("#22c55e").tolist()
    fig = go.Figure(
        go.Bar(
            x=df["ConfidenceFloat"],
            y=df["Condition"],
            orientation="h",
            marker_color=colors,
            text=df["Confidence"],
            textposition="inside",
            hovertemplate="<b>%{y}</b><br>Probability: %{x:.3f}<br>CI: %{customdata}<extra></extra>",
            customdata=df["CI95"],
        )
    )
    fig.add_vline(x=0.5, line_dash="dot", line_color="#14b8a6",
                  annotation_text="Default threshold (0.50)",
                  annotation_font_color="#7bbfb8")
    fig.update_layout(
        title="Pathology Probability Distribution",
        paper_bgcolor="#071c1f",
        plot_bgcolor="#071c1f",
        font=dict(color="#c8e6e2", family="Inter, system-ui, sans-serif"),
        xaxis=dict(title="Probability", range=[0, 1], gridcolor="#123c42"),
        yaxis=dict(title="", gridcolor="#123c42"),
        margin=dict(l=8, r=8, t=40, b=8),
        height=460,
    )
    return fig


def _history_df() -> pd.DataFrame:
    hist = HISTORY.get_history()
    if not hist:
        return pd.DataFrame(columns=["Scan ID", "Timestamp", "Risk Score", "Primary Concern"])
    rows = []
    for h in hist:
        top = max(h["predictions"], key=h["predictions"].get) if h["predictions"] else "—"
        rows.append({
            "Scan ID":       h["scan_id"],
            "Timestamp":     h["timestamp"],
            "Risk Score":    h["risk_score"],
            "Primary Concern": top,
        })
    return pd.DataFrame(rows)


def _trend_chart() -> go.Figure:
    hist = HISTORY.get_history()
    layout = dict(paper_bgcolor="#071c1f", plot_bgcolor="#071c1f",
                  font=dict(color="#c8e6e2"), height=280,
                  margin=dict(l=8, r=8, t=30, b=8))
    if len(hist) < 2:
        fig = go.Figure()
        fig.update_layout(title="Session trend (requires 2+ scans)", **layout)
        return fig
    rows = []
    for scan in hist:
        top = max(scan["predictions"], key=scan["predictions"].get)
        rows.append({"Timestamp": scan["timestamp"],
                     "Probability": scan["predictions"][top],
                     "Pathology": top})
    df = pd.DataFrame(rows)
    fig = px.line(df, x="Timestamp", y="Probability", color="Pathology", markers=True,
                  title="Session Trend")
    fig.update_layout(**layout)
    return fig


def _format_findings_df(predictions: Dict[str, float],
                         cis: Dict[str, float],
                         predictor) -> pd.DataFrame:
    rows = []
    for pathology, prob in predictions.items():
        risk      = predictor.URGENCY.get(pathology, "LOW")
        threshold = predictor.CLINICAL_THRESHOLDS.get(pathology, 0.50)
        action    = "Review" if prob >= threshold else "Observe"
        rows.append({
            "Condition":    pathology,
            "Confidence":   f"{prob * 100:.2f}%",
            "ConfidenceFloat": prob,
            "CI95":         f"±{cis.get(pathology, 0.0):.3f}",
            "Risk":         risk,
            "Risk Level":   _risk_label(risk),
            "Models Agree": "Yes" if cis.get(pathology, 0.0) < 0.15 else "No",
            "Action":       action,
        })
    return pd.DataFrame(rows).sort_values("ConfidenceFloat", ascending=False)


def _load_input(image: Optional[Image.Image],
                dicom_file) -> tuple[np.ndarray | Image.Image, dict]:
    if dicom_file is not None and is_dicom(dicom_file.name):
        dcm = load_dicom(dicom_file.name)
        return dcm["pixel_array"], dcm
    if image is not None:
        return image.convert("RGB"), {}
    raise ValueError("No image provided. Upload a chest X-ray (JPG/PNG) or a DICOM file.")


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze(image: Optional[Image.Image], dicom_file, patient_id: str,
            age: int, sex: str):
    """Run ensemble inference and return all panel outputs."""
    try:
        predictor = get_predictor()
        input_data, dcm_meta = _load_input(image, dicom_file)

        quality_arr = (
            input_data
            if isinstance(input_data, np.ndarray)
            else np.asarray(input_data.convert("L")) / 255.0
        )
        quality = estimate_image_quality(quality_arr)

        preds, cis, agreement, img2d, tensor, model_outputs = (
            predictor.predict_ensemble(input_data)
        )

        # Top-3 pathology indices for GradCAM
        top3 = sorted(preds, key=preds.get, reverse=True)[:3]
        top3_idx = [
            predictor.model_nih.pathologies.index(p)
            for p in top3
            if p in predictor.model_nih.pathologies
        ]
        if not top3_idx:
            top3_idx = [0, 1, 2]

        gradpp = generate_gradcam_plus_plus(predictor.model_nih, tensor, img2d, top3_idx[0])
        grid   = generate_comparison_grid(predictor.model_nih, tensor, img2d, top3_idx[:3])

        scorer = RiskScorer(predictor.URGENCY, predictor.CLINICAL_THRESHOLDS)
        risk   = scorer.calculate_risk_score(preds, patient_age=age, patient_sex=sex)

        second = SECOND_OPINION_ENGINE.generate_second_opinion(preds, cis, agreement)

        patient_info = {
            "patient_id": patient_id or "Unknown",
            "age": age,
            "sex": sex,
            "quality": quality,
            **dcm_meta,
        }
        report = generate_clinical_report(
            preds, cis, agreement, patient_info,
            predictor.PATHOLOGY_DESCRIPTIONS,
            predictor.URGENCY,
            predictor.CLINICAL_THRESHOLDS,
        )

        findings_df = _format_findings_df(preds, cis, predictor)
        chart       = _build_chart(findings_df)
        scan_id     = str(uuid.uuid4())[:8]
        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        HISTORY.add_scan(scan_id, preds, risk["risk_score"], timestamp)

        per_model_rows = [
            {"Model": mname, "Condition": pathology, "Probability": f"{value * 100:.2f}%"}
            for mname, vals in model_outputs.items()
            for pathology, value in vals.items()
        ]
        per_model_df = pd.DataFrame(per_model_rows)

        # Summary line
        if any(preds[p] >= predictor.CLINICAL_THRESHOLDS.get(p, 0.50) for p in preds):
            executive = f"Primary concern: {top3[0]}  ({preds[top3[0]] * 100:.1f}% confidence)"
        else:
            executive = "No findings exceed detection thresholds. No acute concerns identified."

        # Emergency banner
        is_critical = any(
            predictor.URGENCY.get(p) == "CRITICAL"
            and preds[p] >= predictor.CLINICAL_THRESHOLDS.get(p, 0.5)
            for p in preds
        )
        emergency_html = (
            "<div class='alert-critical'><span class='alert-icon'>&#9888;</span> CRITICAL FINDING — IMMEDIATE RADIOLOGIST REVIEW REQUIRED</div>"
            if is_critical else ""
        )

        LAST_ANALYSIS.clear()
        LAST_ANALYSIS.update({"report": report, "heatmap": gradpp,
                               "second": second["second_opinion_text"]})

        display_df = findings_df[["Condition", "Confidence", "CI95",
                                   "Risk Level", "Models Agree", "Action"]]
        history_ids = [s["scan_id"] for s in HISTORY.get_history()]

        risk_text    = f"{risk['overall_risk']}  |  Score: {risk['risk_score']} / 100"
        quality_text = (
            f"{quality['quality_label']}  ({quality['quality_score']} / 100)\n"
            f"Brightness: {quality['brightness']}   "
            f"Noise: {quality['noise_level']}   "
            f"Contrast: {quality['contrast']:.3f}"
        )
        agreement_text = f"Model consensus: {agreement:.1f}%"
        exec_md = f"**Summary:** {executive}"

        return (
            display_df,          # findings table
            chart,               # probability chart
            agreement_text,      # consensus textbox
            gradpp,              # heatmap image
            grid,                # comparison grid
            "GradCAM++ and EigenCAM identify which image regions drove each prediction.",
            report,              # clinical report textbox
            exec_md,             # executive summary markdown
            _history_df(),       # session history table
            _trend_chart(),      # trend plot
            per_model_df,        # per-model breakdown
            risk_text,           # risk sidebar textbox
            second["second_opinion_text"],   # second opinion
            risk["recommended_followup"],    # recommendations
            quality_text,        # image quality
            emergency_html,      # emergency alert
            gr.update(choices=history_ids, value=scan_id),  # history dropdown
        )

    except Exception as exc:
        err_msg = f"Analysis error:\n{exc}\n\n{traceback.format_exc()}"
        empty_df = pd.DataFrame(
            columns=["Condition", "Confidence", "CI95", "Risk Level", "Models Agree", "Action"]
        )
        history_ids = [s["scan_id"] for s in HISTORY.get_history()]
        return (
            empty_df,
            go.Figure().update_layout(title="No data", paper_bgcolor="#071c1f",
                                       plot_bgcolor="#071c1f", font=dict(color="#c8e6e2")),
            "Consensus: —",
            None,
            None,
            "",
            err_msg,
            "",
            _history_df(),
            _trend_chart(),
            pd.DataFrame(columns=["Model", "Condition", "Probability"]),
            "—",
            "Second opinion unavailable.",
            "No recommendation available.",
            "—",
            "",
            gr.update(choices=history_ids, value=None),
        )


# ---------------------------------------------------------------------------
# Button callbacks
# ---------------------------------------------------------------------------

def run_second_opinion() -> str:
    if not LAST_ANALYSIS:
        return "No analysis has been run yet."
    return LAST_ANALYSIS.get("second", "Second opinion unavailable.")


def download_pdf():
    if not LAST_ANALYSIS:
        raise gr.Error("Run an analysis first before downloading the report.")
    pdf_bytes = generate_pdf_report(LAST_ANALYSIS["report"], LAST_ANALYSIS["heatmap"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        return tmp.name


def load_history_entry(scan_id: str) -> str:
    scan = next((s for s in HISTORY.get_history() if s["scan_id"] == scan_id), None)
    if not scan:
        return "No scan selected."
    top = max(scan["predictions"], key=scan["predictions"].get)
    return (f"Scan: {scan_id}   |   Recorded: {scan['timestamp']}\n"
            f"Risk score: {scan['risk_score']} / 100   |   Primary finding: {top}")


# ---------------------------------------------------------------------------
# CSS – World-class premium medical dark theme
# ---------------------------------------------------------------------------

CSS = """
/* ═══════════════════════════════════════════════════════════════════════════
   SKIAVOX — Premium Medical Dark Theme
   CSS Custom Properties / Design Tokens
═══════════════════════════════════════════════════════════════════════════ */

:root {
  /* Background layers — deep clinical teal-dark */
  --bg-base:          #030e0f;
  --bg-card:          #071c1f;
  --bg-elevated:      #0a2529;
  --bg-surface:       #0c2d32;
  --bg-hover:         #0f3840;
  --bg-input:         #04100f;
  --bg-overlay:       rgba(3, 14, 15, 0.85);

  /* Borders */
  --border:           #123c42;
  --border-subtle:    #0a2428;
  --border-bright:    #1e5c66;
  --border-accent:    rgba(20, 184, 166, 0.35);
  --border-glow:      rgba(16, 185, 129, 0.25);

  /* Text */
  --text-primary:     #e4f5f2;
  --text-secondary:   #7bbfb8;
  --text-muted:       #3a7070;
  --text-disabled:    #1e4040;
  --text-label:       #4a9090;

  /* Accent palette — teal / emerald */
  --accent-blue:      #14b8a6;
  --accent-blue-dim:  #0d9488;
  --accent-blue-glow: rgba(20, 184, 166, 0.20);
  --accent-indigo:    #10b981;
  --accent-purple:    #059669;
  --accent-cyan:      #06b6d4;
  --accent-teal:      #0d9488;

  /* Semantic / risk — unchanged (universal medical severity colours) */
  --risk-critical:    #ef4444;
  --risk-critical-bg: rgba(239, 68, 68, 0.12);
  --risk-high:        #f97316;
  --risk-high-bg:     rgba(249, 115, 22, 0.12);
  --risk-moderate:    #eab308;
  --risk-moderate-bg: rgba(234, 179, 8, 0.12);
  --risk-low:         #22c55e;
  --risk-low-bg:      rgba(34, 197, 94, 0.12);

  /* Gradients */
  --grad-primary:     linear-gradient(135deg, #0d9488 0%, #059669 100%);
  --grad-header:      linear-gradient(135deg, #030e0f 0%, #071c1f 50%, #040f10 100%);
  --grad-card:        linear-gradient(160deg, #071c1f 0%, #04100f 100%);
  --grad-accent-line: linear-gradient(90deg, #14b8a6, #10b981, #06b6d4, #0d9488, #14b8a6);
  --grad-section:     linear-gradient(90deg, #14b8a6 0%, transparent 100%);

  /* Shadows */
  --shadow-card:      0 4px 24px rgba(0, 0, 0, 0.6), 0 1px 4px rgba(0,0,0,0.4);
  --shadow-elevated:  0 8px 32px rgba(0, 0, 0, 0.7), 0 2px 8px rgba(0,0,0,0.5);
  --shadow-glow-blue: 0 0 20px rgba(20, 184, 166, 0.25), 0 0 40px rgba(20, 184, 166, 0.10);
  --shadow-glow-btn:  0 4px 20px rgba(13, 148, 136, 0.50), 0 2px 8px rgba(20, 184, 166, 0.30);

  /* Radii */
  --radius-sm:   6px;
  --radius-md:   10px;
  --radius-lg:   14px;
  --radius-pill: 999px;

  /* Transitions */
  --transition:  0.2s ease;
  --transition-slow: 0.35s ease;
}

/* ── Reset & Base ────────────────────────────────────────────────────────── */

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

html {
  scroll-behavior: smooth;
}

body,
.gradio-container {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', 'SF Pro Display', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */

::-webkit-scrollbar {
  width: 4px;
  height: 4px;
}
::-webkit-scrollbar-track {
  background: var(--bg-base);
  border-radius: var(--radius-pill);
}
::-webkit-scrollbar-thumb {
  background: var(--border-bright);
  border-radius: var(--radius-pill);
  transition: background var(--transition);
}
::-webkit-scrollbar-thumb:hover {
  background: var(--accent-blue);
}

/* ── Animated Rainbow Header Bar ────────────────────────────────────────── */

@keyframes shimmer-rainbow {
  0%   { background-position: 0% 50%;   }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%;   }
}

.ms-rainbow-bar {
  height: 3px;
  width: 100%;
  background: var(--grad-accent-line);
  background-size: 300% 300%;
  animation: shimmer-rainbow 4s ease infinite;
  border: none;
  margin: 0;
}

/* ── Header ─────────────────────────────────────────────────────────────── */

.ms-header-wrap {
  background: var(--grad-header);
  border-bottom: 1px solid var(--border);
  padding: 22px 28px 18px;
  position: relative;
  overflow: hidden;
}

.ms-header-wrap::before {
  content: '';
  position: absolute;
  top: -60px;
  right: -60px;
  width: 280px;
  height: 280px;
  background: radial-gradient(circle, rgba(20,184,166,0.07) 0%, transparent 70%);
  pointer-events: none;
}

.ms-header-wrap::after {
  content: '';
  position: absolute;
  bottom: -80px;
  left: 30%;
  width: 200px;
  height: 200px;
  background: radial-gradient(circle, rgba(16,185,129,0.05) 0%, transparent 70%);
  pointer-events: none;
}

.ms-header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  max-width: 100%;
  position: relative;
  z-index: 1;
}

.ms-logo-group {
  display: flex;
  align-items: center;
  gap: 14px;
}

.ms-logo-icon {
  width: 44px;
  height: 44px;
  background: linear-gradient(135deg, rgba(20,184,166,0.15) 0%, rgba(16,185,129,0.15) 100%);
  border: 1px solid rgba(20,184,166,0.35);
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0 16px rgba(20,184,166,0.20);
  flex-shrink: 0;
}

.ms-logo-text h1 {
  font-size: 22px !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
  line-height: 1.1 !important;
  color: #f0f6ff !important;
  margin: 0 !important;
}

.ms-logo-text h1 .brand-accent {
  background: linear-gradient(90deg, #14b8a6, #34d399, #059669);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.ms-center-text {
  text-align: center;
  flex: 1;
  padding: 0 24px;
}

.ms-center-text .subtitle {
  font-size: 12px !important;
  color: var(--text-secondary) !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  font-weight: 500 !important;
}

.ms-status-pills {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
}

/* ── Status Pill (animated pulse dot) ───────────────────────────────────── */

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1);   }
  50%       { opacity: 0.5; transform: scale(0.75); }
}

.ms-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  background: rgba(10, 22, 40, 0.85);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-pill);
  font-size: 11px !important;
  font-weight: 600 !important;
  color: var(--accent-blue) !important;
  letter-spacing: 0.05em !important;
  white-space: nowrap;
  backdrop-filter: blur(8px);
}

.ms-pill-dot {
  width: 7px;
  height: 7px;
  background: var(--risk-low);
  border-radius: 50%;
  flex-shrink: 0;
  animation: pulse-dot 2s ease-in-out infinite;
  box-shadow: 0 0 6px rgba(34, 197, 94, 0.6);
}

/* ── Section Headings ────────────────────────────────────────────────────── */

.ms-section-heading {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-subtle);
}

.ms-section-heading::before {
  content: '';
  display: block;
  width: 3px;
  height: 18px;
  background: var(--grad-primary);
  border-radius: var(--radius-pill);
  flex-shrink: 0;
}

.ms-section-heading span {
  font-size: 11px !important;
  font-weight: 700 !important;
  color: var(--text-secondary) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.10em !important;
}

/* ── Section Divider ─────────────────────────────────────────────────────── */

.ms-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-accent), transparent);
  margin: 14px 0;
  border: none;
}

/* ── Card Panels ─────────────────────────────────────────────────────────── */

.gradio-container .block,
.gradio-container .form {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow-card) !important;
  transition: border-color var(--transition) !important;
}

.gradio-container .block:hover,
.gradio-container .form:hover {
  border-color: var(--border-bright) !important;
}

/* ── Left panel card wrapper ─────────────────────────────────────────────── */

.ms-controls-card {
  background: var(--grad-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px 16px 14px;
  box-shadow: var(--shadow-card);
  height: 100%;
}

/* ── Right sidebar card ──────────────────────────────────────────────────── */

.ms-sidebar-section {
  background: linear-gradient(160deg, var(--bg-card) 0%, var(--bg-base) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 14px 14px 12px;
  margin-bottom: 10px;
  box-shadow: var(--shadow-card);
  transition: border-color var(--transition), box-shadow var(--transition);
}

.ms-sidebar-section:hover {
  border-color: var(--border-accent);
  box-shadow: var(--shadow-card), 0 0 12px var(--accent-blue-glow);
}

/* ── Labels ──────────────────────────────────────────────────────────────── */

.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .label-wrap > span {
  color: var(--text-label) !important;
  font-size: 10.5px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  font-weight: 700 !important;
}

/* ── Inputs, Textareas, Selects ──────────────────────────────────────────── */

.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container input[type="email"],
.gradio-container textarea,
.gradio-container select {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 13px !important;
  font-family: inherit !important;
  transition: border-color var(--transition), box-shadow var(--transition) !important;
  padding: 8px 12px !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 3px var(--accent-blue-glow), 0 0 12px rgba(20,184,166,0.15) !important;
  outline: none !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--text-muted) !important;
}

/* ── Slider ──────────────────────────────────────────────────────────────── */

.gradio-container input[type="range"] {
  accent-color: var(--accent-blue) !important;
  height: 4px !important;
}

.gradio-container .wrap.svelte-1oiin9d {
  background: var(--bg-input) !important;
}

/* ── Radio Buttons ───────────────────────────────────────────────────────── */

.gradio-container .wrap.svelte-1cl284s input[type="radio"] {
  accent-color: var(--accent-blue) !important;
}

/* ── Buttons — Primary ───────────────────────────────────────────────────── */

.gradio-container button.primary,
button.primary {
  background: var(--grad-primary) !important;
  border: none !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  letter-spacing: 0.04em !important;
  border-radius: var(--radius-md) !important;
  padding: 12px 24px !important;
  box-shadow: var(--shadow-glow-btn) !important;
  transition: transform var(--transition), box-shadow var(--transition), filter var(--transition) !important;
  cursor: pointer !important;
  position: relative !important;
  overflow: hidden !important;
}

.gradio-container button.primary::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  transition: left 0.5s ease;
}

.gradio-container button.primary:hover {
  transform: translateY(-1px) scale(1.01) !important;
  box-shadow: 0 6px 28px rgba(13,148,136,0.55), 0 3px 12px rgba(20,184,166,0.40) !important;
  filter: brightness(1.08) !important;
}

.gradio-container button.primary:hover::before {
  left: 100%;
}

.gradio-container button.primary:active {
  transform: translateY(0) scale(0.99) !important;
}

/* ── Buttons — Secondary ─────────────────────────────────────────────────── */

.gradio-container button.secondary,
button.secondary {
  background: transparent !important;
  border: 1px solid var(--border-bright) !important;
  color: var(--text-secondary) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  letter-spacing: 0.04em !important;
  border-radius: var(--radius-sm) !important;
  padding: 8px 14px !important;
  transition: background var(--transition), border-color var(--transition),
              color var(--transition), box-shadow var(--transition) !important;
  cursor: pointer !important;
}

.gradio-container button.secondary:hover {
  background: var(--bg-elevated) !important;
  border-color: var(--accent-blue) !important;
  color: var(--text-primary) !important;
  box-shadow: 0 0 12px rgba(20,184,166,0.20) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */

.gradio-container .tab-nav {
  background: var(--bg-card) !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 8px !important;
  gap: 4px !important;
  display: flex !important;
  align-items: center !important;
}

.gradio-container .tab-nav button {
  background: transparent !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  padding: 8px 16px !important;
  transition: color var(--transition), background var(--transition) !important;
  position: relative !important;
  cursor: pointer !important;
  margin: 4px 0 !important;
}

.gradio-container .tab-nav button::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 8px;
  right: 8px;
  height: 2px;
  background: var(--grad-primary);
  border-radius: var(--radius-pill);
  transform: scaleX(0);
  transition: transform var(--transition);
}

.gradio-container .tab-nav button:hover {
  background: var(--bg-elevated) !important;
  color: var(--text-secondary) !important;
}

.gradio-container .tab-nav button.selected {
  background: rgba(20,184,166,0.08) !important;
  color: var(--accent-blue) !important;
}

.gradio-container .tab-nav button.selected::after {
  transform: scaleX(1);
}

/* ── DataFrames / Tables ─────────────────────────────────────────────────── */

.gradio-container table {
  background: var(--bg-card) !important;
  border-collapse: collapse !important;
  font-size: 12.5px !important;
  width: 100% !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
}

.gradio-container thead {
  position: sticky !important;
  top: 0 !important;
  z-index: 2 !important;
}

.gradio-container th {
  background: var(--bg-elevated) !important;
  color: var(--text-label) !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.10em !important;
  padding: 10px 14px !important;
  border-bottom: 1px solid var(--border) !important;
  white-space: nowrap !important;
}

.gradio-container td {
  color: var(--text-primary) !important;
  padding: 8px 14px !important;
  border-bottom: 1px solid var(--border-subtle) !important;
  transition: background var(--transition) !important;
  vertical-align: middle !important;
}

.gradio-container tr:nth-child(even) td {
  background: rgba(10, 22, 40, 0.4) !important;
}

.gradio-container tr:hover td {
  background: var(--bg-hover) !important;
}

/* ── Upload Zone ─────────────────────────────────────────────────────────── */

.gradio-container .upload-container,
.gradio-container [data-testid="image"],
.gradio-container .image-upload-container {
  background: var(--bg-input) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius-md) !important;
  transition: border-color var(--transition), background var(--transition) !important;
}

.gradio-container .upload-container:hover,
.gradio-container [data-testid="image"]:hover {
  border-color: var(--accent-blue) !important;
  background: rgba(20,184,166,0.04) !important;
}

/* ── Alert Critical ──────────────────────────────────────────────────────── */

@keyframes critical-pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4), 0 0 20px rgba(239,68,68,0.2);
    border-color: var(--risk-critical);
  }
  50% {
    box-shadow: 0 0 0 6px rgba(239, 68, 68, 0), 0 0 32px rgba(239,68,68,0.35);
    border-color: rgba(239, 68, 68, 0.6);
  }
}

.alert-critical {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 14px 18px;
  background: var(--risk-critical-bg);
  border: 1.5px solid var(--risk-critical);
  border-radius: var(--radius-md);
  color: #fca5a5;
  font-weight: 800;
  font-size: 12.5px;
  text-align: center;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  animation: critical-pulse 1.8s ease-in-out infinite;
  margin-bottom: 10px;
}

.alert-icon {
  font-size: 16px;
  flex-shrink: 0;
}

/* ── Risk Badge Pills ────────────────────────────────────────────────────── */

.risk-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: var(--radius-pill);
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  line-height: 1.4 !important;
}

.risk-badge.critical {
  background: var(--risk-critical-bg);
  color: var(--risk-critical);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.risk-badge.high {
  background: var(--risk-high-bg);
  color: var(--risk-high);
  border: 1px solid rgba(249, 115, 22, 0.3);
}

.risk-badge.moderate {
  background: var(--risk-moderate-bg);
  color: var(--risk-moderate);
  border: 1px solid rgba(234, 179, 8, 0.3);
}

.risk-badge.low {
  background: var(--risk-low-bg);
  color: var(--risk-low);
  border: 1px solid rgba(34, 197, 94, 0.3);
}

/* ── Sidebar Cards (right panel) ─────────────────────────────────────────── */

.ms-risk-card {
  background: linear-gradient(160deg, var(--bg-card) 0%, #060f20 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 14px;
  margin-bottom: 8px;
  transition: border-color var(--transition), box-shadow var(--transition);
}

.ms-risk-card:hover {
  border-color: var(--border-accent);
  box-shadow: 0 0 16px var(--accent-blue-glow);
}

.ms-card-label {
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: var(--text-muted) !important;
  margin-bottom: 8px !important;
  display: flex !important;
  align-items: center !important;
  gap: 7px !important;
}

.ms-card-label-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent-blue);
  box-shadow: 0 0 6px var(--accent-blue);
  flex-shrink: 0;
}

.ms-card-label-dot.orange { background: var(--risk-high); box-shadow: 0 0 6px var(--risk-high); }
.ms-card-label-dot.green  { background: var(--risk-low);  box-shadow: 0 0 6px var(--risk-low);  }
.ms-card-label-dot.yellow { background: var(--risk-moderate); box-shadow: 0 0 6px var(--risk-moderate); }

/* ── Gradio Textbox overrides ─────────────────────────────────────────────── */

.gradio-container .gr-box,
.gradio-container .box {
  background: var(--bg-card) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-md) !important;
}

/* ── Row / Group overrides ───────────────────────────────────────────────── */

.gradio-container .gr-group,
.gradio-container .group {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 12px !important;
}

/* ── Dropdown ────────────────────────────────────────────────────────────── */

.gradio-container .dropdown,
.gradio-container [data-testid="dropdown"] {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
}

.gradio-container .dropdown:focus-within {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 3px var(--accent-blue-glow) !important;
}

/* ── Plotly chart container ──────────────────────────────────────────────── */

.gradio-container .js-plotly-plot,
.gradio-container .plotly {
  background: transparent !important;
  border-radius: var(--radius-md) !important;
}

/* ── Image viewer ────────────────────────────────────────────────────────── */

.gradio-container .image-container img {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border) !important;
}

/* ── Disclaimer text ─────────────────────────────────────────────────────── */

.ms-disclaimer {
  font-size: 10px !important;
  color: var(--text-muted) !important;
  line-height: 1.6 !important;
  padding: 10px 0 0 !important;
  border-top: 1px solid var(--border-subtle) !important;
  margin-top: 10px !important;
}

/* ── Model info card ─────────────────────────────────────────────────────── */

.ms-model-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 16px;
  margin-bottom: 12px;
  transition: border-color var(--transition);
}

.ms-model-card:hover {
  border-color: var(--border-accent);
}

.ms-model-name {
  font-size: 13px !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  margin-bottom: 6px !important;
  letter-spacing: -0.01em !important;
}

.ms-model-meta {
  font-size: 11.5px !important;
  color: var(--text-secondary) !important;
  line-height: 1.7 !important;
}

.ms-model-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 9px;
  background: rgba(20,184,166,0.10);
  border: 1px solid rgba(20,184,166,0.25);
  border-radius: var(--radius-pill);
  font-size: 10px;
  font-weight: 600;
  color: var(--accent-blue);
  letter-spacing: 0.04em;
  margin-top: 6px;
}

/* ── Gradient section divider line ───────────────────────────────────────── */

.ms-gradient-line {
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, var(--border-accent) 30%, var(--border-glow) 70%, transparent 100%);
  margin: 16px 0;
  border: none;
}

/* ── Tab content padding ─────────────────────────────────────────────────── */

.gradio-container .tabitem {
  padding: 16px 4px 4px !important;
}

/* ── Markdown ────────────────────────────────────────────────────────────── */

.gradio-container .prose,
.gradio-container .md {
  color: var(--text-primary) !important;
  font-size: 13px !important;
}

.gradio-container .prose strong,
.gradio-container .md strong {
  color: var(--accent-blue) !important;
}

/* ── File upload component ───────────────────────────────────────────────── */

.gradio-container .file-preview {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
}

/* ── Loading / spinner ───────────────────────────────────────────────────── */

.gradio-container .loading {
  color: var(--accent-blue) !important;
}

/* ── Gradio footer (hide) ────────────────────────────────────────────────── */

footer,
.gradio-container footer,
.footer {
  display: none !important;
}

/* ── Top padding of entire container ────────────────────────────────────── */

.gradio-container > .main {
  padding-top: 0 !important;
}

/* ── Consensus / metadata pills inline ──────────────────────────────────── */

.ms-meta-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.ms-meta-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-pill);
  font-size: 11px;
  color: var(--text-secondary);
  letter-spacing: 0.04em;
}

/* ── Number / value emphasis ─────────────────────────────────────────────── */

.ms-value-em {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  line-height: 1.2;
}

.ms-value-sub {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 2px;
}

/* ── Gradio accordion ────────────────────────────────────────────────────── */

.gradio-container .accordion {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}

/* ── Image label overlay ─────────────────────────────────────────────────── */

.gradio-container .image-label {
  background: rgba(5, 10, 20, 0.75) !important;
  color: var(--text-secondary) !important;
  font-size: 11px !important;
  backdrop-filter: blur(4px) !important;
}

/* ── Ensure consistent row spacing ──────────────────────────────────────── */

.gradio-container .row {
  gap: 12px !important;
}

/* ── Responsive tweaks ───────────────────────────────────────────────────── */

@media (max-width: 1024px) {
  .ms-header-inner {
    flex-direction: column;
    gap: 14px;
    text-align: center;
  }
  .ms-center-text {
    padding: 0;
  }
  .ms-status-pills {
    justify-content: center;
  }
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div class="ms-rainbow-bar"></div>
<div class="ms-header-wrap">
  <div class="ms-header-inner">

    <!-- Left: Logo -->
    <div class="ms-logo-group">
      <div class="ms-logo-icon">
        <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="11" y="3" width="4" height="20" rx="2" fill="url(#cross-grad)"/>
          <rect x="3" y="11" width="20" height="4" rx="2" fill="url(#cross-grad)"/>
          <defs>
            <linearGradient id="cross-grad" x1="3" y1="3" x2="23" y2="23" gradientUnits="userSpaceOnUse">
              <stop stop-color="#14b8a6"/>
              <stop offset="1" stop-color="#34d399"/>
            </linearGradient>
          </defs>
        </svg>
      </div>
      <div class="ms-logo-text">
        <h1>Skia<span class="brand-accent">vox</span></h1>
      </div>
    </div>

    <!-- Center: Subtitle -->
    <div class="ms-center-text">
      <div class="subtitle">Advanced Diagnostic Imaging Platform</div>
    </div>

    <!-- Right: Status pills -->
    <div class="ms-status-pills">
      <span class="ms-pill">
        <span class="ms-pill-dot"></span>
        3 Models Active
      </span>
      <span class="ms-pill">
        <span class="ms-pill-dot"></span>
        Ensemble Mode
      </span>
      <span class="ms-pill">
        <span class="ms-pill-dot"></span>
        Research v2.0
      </span>
    </div>

  </div>
</div>
<div class="ms-rainbow-bar"></div>
"""

MODEL_DETAILS_HTML = """
<div style="padding: 4px 0;">

  <div class="ms-model-card">
    <div class="ms-model-name">DenseNet121 — NIH ChestX-ray14</div>
    <div class="ms-model-meta">
      Input resolution: 224 &times; 224 &nbsp;&bull;&nbsp;
      Training set: 112,120 frontal-view images<br>
      Pathology labels: 14 &nbsp;&bull;&nbsp; Ensemble weight: <strong style="color:#e8f0fe;">0.40</strong>
    </div>
    <span class="ms-model-badge">Primary Model</span>
  </div>

  <div class="ms-model-card">
    <div class="ms-model-name">DenseNet121 — CheXpert</div>
    <div class="ms-model-meta">
      Input resolution: 224 &times; 224 &nbsp;&bull;&nbsp;
      Training set: 224,316 chest radiographs<br>
      Pathology labels: 14 &nbsp;&bull;&nbsp; Ensemble weight: <strong style="color:#e8f0fe;">0.35</strong>
    </div>
    <span class="ms-model-badge">Secondary Model</span>
  </div>

  <div class="ms-model-card">
    <div class="ms-model-name">ResNet50 — Multi-Source Combined</div>
    <div class="ms-model-meta">
      Input resolution: 512 &times; 512 &nbsp;&bull;&nbsp;
      Training set: Multi-dataset combined corpus<br>
      Pathology labels: variable &nbsp;&bull;&nbsp; Ensemble weight: <strong style="color:#e8f0fe;">0.25</strong>
    </div>
    <span class="ms-model-badge">Tertiary Model</span>
  </div>

  <div class="ms-gradient-line"></div>

  <div style="font-size:12px; color: var(--text-secondary); line-height: 1.8; padding: 4px 0;">
    Final prediction = weighted average of sigmoid outputs across all three models.<br>
    Confidence intervals derived from model disagreement (cross-ensemble variance).<br>
    GradCAM++ saliency maps computed on the NIH DenseNet121 backbone.
  </div>

</div>
"""


with gr.Blocks(css=CSS, title="Skiavox") as demo:

    # Header
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        # ── LEFT: controls ──────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=270):

            gr.HTML("""
            <div class="ms-section-heading" style="margin-bottom:14px;">
              <span>Patient &amp; Imaging</span>
            </div>
            """)

            image_input = gr.Image(
                type="pil", height=240, label="Chest X-ray Image (JPG / PNG)",
                sources=["upload"],
            )
            dicom_input = gr.File(
                label="DICOM File  —  optional (.dcm / .dicom)",
                file_types=[".dcm", ".dicom"],
            )

            gr.HTML('<div class="ms-gradient-line"></div>')

            gr.HTML("""
            <div class="ms-section-heading">
              <span>Patient Demographics</span>
            </div>
            """)

            with gr.Group():
                patient_id = gr.Textbox(
                    label="Patient ID", placeholder="Optional identifier", max_lines=1
                )
                age = gr.Slider(0, 100, value=45, step=1, label="Patient Age (years)")
                sex = gr.Radio(
                    ["Male", "Female", "Unknown"], value="Unknown", label="Biological Sex"
                )

            gr.HTML('<div class="ms-gradient-line"></div>')

            analyze_btn = gr.Button(
                "Run Analysis", variant="primary", size="lg"
            )

            with gr.Row():
                second_btn = gr.Button("Second Opinion", variant="secondary", size="sm")
                pdf_btn    = gr.Button("Download PDF",   variant="secondary", size="sm")

            gr.HTML('<div class="ms-gradient-line"></div>')

            gr.HTML("""
            <div class="ms-section-heading">
              <span>Session History</span>
            </div>
            """)

            history_dropdown = gr.Dropdown(
                label="Select Prior Scan", choices=[], interactive=True
            )
            history_note = gr.Textbox(
                label="Scan Summary", value="", interactive=False, lines=2
            )

            gr.HTML("""
            <div class="ms-disclaimer">
              Research prototype only. Not FDA-cleared or CE-marked.<br>
              Not intended for clinical diagnosis or treatment decisions.<br>
              Always consult a licensed radiologist or physician.
            </div>
            """)

        # ── MIDDLE: analysis results ─────────────────────────────────────────
        with gr.Column(scale=5, min_width=500):

            with gr.Tab("AI Findings"):
                findings_df = gr.DataFrame(
                    headers=["Condition", "Confidence", "CI95",
                              "Risk Level", "Models Agree", "Action"],
                    label="Detection Results",
                    wrap=True,
                    interactive=False,
                )
                prob_plot = gr.Plot(label="Pathology Probability Distribution")
                agreement_text = gr.Textbox(
                    label="Model Consensus", value="", interactive=False, lines=1
                )

            with gr.Tab("Explainability"):
                heatmap_img    = gr.Image(height=320, label="GradCAM++ Saliency Heatmap")
                comparison_img = gr.Image(height=320, label="Multi-Pathology Comparison Grid")
                explain_text   = gr.Textbox(
                    label="Interpretation Note", value="", interactive=False, lines=2
                )

            with gr.Tab("Clinical Report"):
                report_box   = gr.Textbox(
                    label="Full Clinical Report", lines=30,
                    value="", interactive=False,
                )
                exec_summary = gr.Markdown(value="")

            with gr.Tab("Session History"):
                hist_table  = gr.DataFrame(
                    label="Prior Scans This Session", interactive=False
                )
                trend_chart = gr.Plot(label="Pathology Probability Trend")

            with gr.Tab("Model Details"):
                gr.HTML(MODEL_DETAILS_HTML)
                per_model_df = gr.DataFrame(
                    label="Per-Model Raw Predictions", interactive=False
                )

        # ── RIGHT: risk panel ────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=260):

            emergency_html = gr.HTML(value="")

            gr.HTML("""
            <div class="ms-card-label">
              <span class="ms-card-label-dot"></span>
              Risk Assessment
            </div>
            """)
            risk_text = gr.Textbox(
                label="Overall Risk  |  Score", value="", interactive=False, lines=2
            )

            gr.HTML('<div class="ms-gradient-line"></div>')

            gr.HTML("""
            <div class="ms-card-label">
              <span class="ms-card-label-dot orange"></span>
              AI Second Opinion
            </div>
            """)
            second_panel = gr.Textbox(
                label="Second Opinion Analysis", value="", interactive=False, lines=9
            )

            gr.HTML('<div class="ms-gradient-line"></div>')

            gr.HTML("""
            <div class="ms-card-label">
              <span class="ms-card-label-dot green"></span>
              Recommendations
            </div>
            """)
            rec_panel = gr.Textbox(
                label="Clinical Follow-up Recommendations", value="", interactive=False, lines=5
            )

            gr.HTML('<div class="ms-gradient-line"></div>')

            gr.HTML("""
            <div class="ms-card-label">
              <span class="ms-card-label-dot yellow"></span>
              Image Quality
            </div>
            """)
            quality_text = gr.Textbox(
                label="Quality Metrics", value="", interactive=False, lines=3
            )

            gr.HTML('<div class="ms-gradient-line"></div>')

            pdf_file = gr.File(label="Generated PDF Report")

    # ── Event wiring ──────────────────────────────────────────────────────────

    analyze_btn.click(
        fn=analyze,
        inputs=[image_input, dicom_input, patient_id, age, sex],
        outputs=[
            findings_df,
            prob_plot,
            agreement_text,
            heatmap_img,
            comparison_img,
            explain_text,
            report_box,
            exec_summary,
            hist_table,
            trend_chart,
            per_model_df,
            risk_text,
            second_panel,
            rec_panel,
            quality_text,
            emergency_html,
            history_dropdown,
        ],
    )

    second_btn.click(fn=run_second_opinion, inputs=None, outputs=second_panel)
    pdf_btn.click(fn=download_pdf, inputs=None, outputs=pdf_file)
    history_dropdown.change(fn=load_history_entry, inputs=history_dropdown, outputs=history_note)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[Skiavox] Pre-loading models before serving...")
    get_predictor()
    print("[Skiavox] Launching at http://0.0.0.0:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
