"""Comprehensive DICOM utilities for Skiavox."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pydicom


def is_dicom(filepath: str) -> bool:
    """Return True if filepath looks like a DICOM file by extension."""
    return Path(filepath).suffix.lower() in {".dcm", ".dicom"}


def load_dicom(filepath: str) -> Dict[str, Any]:
    """Load DICOM image, normalize pixels, and extract key metadata."""
    ds = pydicom.dcmread(filepath)
    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept

    pmin, pmax = float(np.min(pixels)), float(np.max(pixels))
    if (pmax - pmin) < 1e-8:
        norm = np.zeros_like(pixels, dtype=np.float32)
    else:
        norm = ((pixels - pmin) / (pmax - pmin)).astype(np.float32)

    patient_age = str(getattr(ds, "PatientAge", "Unknown"))
    patient_sex = str(getattr(ds, "PatientSex", "Unknown"))
    study_date = str(getattr(ds, "StudyDate", "Unknown"))
    modality = str(getattr(ds, "Modality", "Unknown"))
    manufacturer = str(getattr(ds, "Manufacturer", "Unknown"))
    kvp = str(getattr(ds, "KVP", "Unknown"))
    exposure = str(getattr(ds, "Exposure", "Unknown"))

    return {
        "pixel_array": norm,
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "study_date": study_date,
        "modality": modality,
        "manufacturer": manufacturer,
        "kvp": kvp,
        "exposure": exposure,
        "image_size": tuple(norm.shape),
        "has_metadata": any(
            value != "Unknown"
            for value in [patient_age, patient_sex, study_date, modality, manufacturer]
        ),
    }


def estimate_image_quality(pixel_array: np.ndarray) -> Dict[str, Any]:
    """Estimate coarse image quality metrics from normalized grayscale image."""
    arr = pixel_array.astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    contrast = float(np.std(arr))
    mean_val = float(np.mean(arr))
    noise_proxy = float(np.std(np.diff(arr, axis=0))) if arr.ndim == 2 else contrast

    if mean_val < 0.30:
        brightness = "dark"
    elif mean_val > 0.70:
        brightness = "bright"
    else:
        brightness = "normal"

    if noise_proxy < 0.08:
        noise_level = "low"
    elif noise_proxy < 0.16:
        noise_level = "medium"
    else:
        noise_level = "high"

    score = int(np.clip(100 - (abs(mean_val - 0.5) * 80 + noise_proxy * 120), 0, 100))
    if score >= 85:
        quality_label = "Excellent"
    elif score >= 70:
        quality_label = "Good"
    elif score >= 55:
        quality_label = "Fair"
    else:
        quality_label = "Poor"

    return {
        "contrast": round(contrast, 4),
        "brightness": brightness,
        "noise_level": noise_level,
        "quality_score": score,
        "quality_label": quality_label,
    }
