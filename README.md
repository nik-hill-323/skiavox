# Skiavox — Advanced Diagnostic Imaging Platform

An open-source AI-powered chest X-ray analysis platform using a 3-model ensemble for pathology detection, explainability, and clinical reporting.

> **Research prototype only. Not FDA-cleared. Not intended for clinical diagnosis or treatment decisions. Always consult a licensed radiologist or physician.**

---

## Features

- **Ensemble Inference** — 3 pre-trained models vote on 14+ pathologies with weighted averaging
- **GradCAM++ Explainability** — saliency heatmaps show which regions drove each prediction
- **Risk Stratification** — Critical / High / Moderate / Low with clinical thresholds
- **Confidence Intervals** — cross-model variance quantifies uncertainty
- **DICOM Support** — upload `.dcm` / `.dicom` files directly
- **Clinical PDF Reports** — auto-generated, downloadable
- **Session History & Trends** — track scans across a session
- **Second Opinion Engine** — AI-generated alternative interpretation

---

## Models

| Model | Dataset | Resolution | Weight |
|-------|---------|------------|--------|
| DenseNet121 | NIH ChestX-ray14 (112,120 images) | 224×224 | 0.40 |
| DenseNet121 | CheXpert (224,316 images) | 224×224 | 0.35 |
| ResNet50 | Multi-source combined | 512×512 | 0.25 |

---

## Quickstart

```bash
git clone https://github.com/nik-hill-323/skiavox.git
cd skiavox
pip install -r requirements.txt
python app.py
```

Open **http://localhost:7860** in your browser.

---

## Project Structure

```
skiavox/
├── app.py               # Gradio UI + main application
├── model.py             # Ensemble predictor (3 models)
├── gradcam_utils.py     # GradCAM++ & EigenCAM saliency
├── dicom_handler.py     # DICOM loading & image quality
├── risk_scorer.py       # Risk stratification logic
├── report_generator.py  # Clinical text & PDF reports
├── second_opinion.py    # Second opinion engine
├── patient_history.py   # Session history tracking
├── requirements.txt
└── setup.sh
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full list

---

## Tech Stack

`PyTorch` · `torchxrayvision` · `Gradio` · `GradCAM` · `pydicom` · `Plotly` · `OpenCV` · `ReportLab`

---

## License

MIT License — free to use, modify, and distribute.
