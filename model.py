"""Multi-model ensemble predictor for Skiavox."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from skimage import color, transform


class EnsembleChestPredictor:
    """Run chest X-ray inference with a weighted 3-model ensemble."""

    MODEL_WEIGHTS = [0.4, 0.35, 0.25]

    PATHOLOGY_DESCRIPTIONS = {
        "Atelectasis": "Partial collapse of lung tissue",
        "Consolidation": "Lung tissue filled with inflammatory exudate",
        "Infiltration": "Diffuse abnormal opacity infiltration",
        "Pneumothorax": "Air in pleural space; potential emergency",
        "Edema": "Pulmonary fluid accumulation",
        "Emphysema": "Alveolar destruction and hyperinflation",
        "Fibrosis": "Chronic lung scarring",
        "Effusion": "Pleural fluid accumulation",
        "Pneumonia": "Infectious inflammatory lung process",
        "Pleural_Thickening": "Pleural lining thickening",
        "Cardiomegaly": "Enlarged cardiomediastinal silhouette",
        "Nodule": "Small focal pulmonary opacity",
        "Mass": "Large focal lesion requiring urgent workup",
        "Hernia": "Diaphragmatic/hiatal herniation finding",
        "Lung Lesion": "Pulmonary lesion with malignant potential",
        "Fracture": "Osseous discontinuity suspicious for fracture",
        "Lung Opacity": "Nonspecific increased lung attenuation",
        "Pleural Other": "Other pleural abnormality pattern",
    }

    URGENCY = {
        "Pneumothorax": "CRITICAL",
        "Mass": "HIGH",
        "Pneumonia": "HIGH",
        "Edema": "HIGH",
        "Lung Lesion": "HIGH",
        "Effusion": "MODERATE",
        "Consolidation": "MODERATE",
        "Atelectasis": "MODERATE",
        "Cardiomegaly": "MODERATE",
        "Nodule": "MODERATE",
        "Lung Opacity": "MODERATE",
        "Infiltration": "LOW",
        "Emphysema": "LOW",
        "Fibrosis": "LOW",
        "Pleural_Thickening": "LOW",
        "Pleural Other": "LOW",
        "Fracture": "HIGH",
        "Hernia": "LOW",
    }

    CLINICAL_THRESHOLDS = {
        "Pneumothorax": 0.35,
        "Mass": 0.40,
        "Pneumonia": 0.45,
        "Edema": 0.45,
    }

    def __init__(self) -> None:
        print("[Skiavox] Loading model 1/3: DenseNet121 NIH...")
        self.model_nih = xrv.models.DenseNet(weights="densenet121-res224-nih")
        self.model_nih.eval()

        print("[Skiavox] Loading model 2/3: DenseNet121 CheX...")
        self.model_chex = xrv.models.DenseNet(weights="densenet121-res224-chex")
        self.model_chex.eval()

        print("[Skiavox] Loading model 3/3: ResNet50 ALL...")
        self.model_resnet = xrv.models.ResNet(weights="resnet50-res512-all")
        self.model_resnet.eval()

        self.models = {
            "nih": self.model_nih,
            "chex": self.model_chex,
            "resnet": self.model_resnet,
        }
        print("[Skiavox] All models loaded successfully.")

    def _to_grayscale(self, image_input: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, Image.Image):
            img = np.asarray(image_input).astype(np.float32)
        elif isinstance(image_input, np.ndarray):
            img = image_input.astype(np.float32)
        else:
            raise TypeError("image_input must be PIL.Image or numpy.ndarray")

        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = color.rgb2gray(img)
        elif img.ndim != 2:
            raise ValueError("Input must be 2D grayscale or 3-channel RGB/RGBA")

        if img.max() > 1.0:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def preprocess_image(
        self, image_input: Union[Image.Image, np.ndarray], model_type: str
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Preprocess image to model-specific resolution and tensor shape."""
        base = self._to_grayscale(image_input)
        target_size = 224 if model_type in {"nih", "chex"} else 512
        resized = transform.resize(
            base, (target_size, target_size), anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        normalized = xrv.datasets.normalize(resized, maxval=1.0, reshape=True)
        tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        return resized, tensor

    def _collect_union_pathologies(self) -> list[str]:
        names = []
        for model in self.models.values():
            for p in model.pathologies:
                if p not in names:
                    names.append(p)
        return names

    def predict_ensemble(
        self, image_input: Union[Image.Image, np.ndarray]
    ) -> Tuple[Dict[str, float], Dict[str, float], float, np.ndarray, torch.Tensor, Dict[str, Dict[str, float]]]:
        """Run 3-model ensemble and return weighted predictions + uncertainty metrics."""
        model_outputs: Dict[str, Dict[str, float]] = {}
        all_pathologies = self._collect_union_pathologies()
        resized_224 = None
        tensor_224 = None

        with torch.no_grad():
            for key, model in self.models.items():
                arr, tensor = self.preprocess_image(image_input, key)
                if key == "nih":
                    resized_224, tensor_224 = arr, tensor
                probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().numpy().tolist()
                model_outputs[key] = {
                    pathology: float(prob)
                    for pathology, prob in zip(model.pathologies, probs)
                }

        weighted_predictions: Dict[str, float] = {}
        confidence_intervals: Dict[str, float] = {}
        agreement_counter = 0
        total_counter = 0

        for pathology in all_pathologies:
            values = []
            for model_name in ["nih", "chex", "resnet"]:
                values.append(model_outputs[model_name].get(pathology, 0.0))
            weighted = (
                values[0] * self.MODEL_WEIGHTS[0]
                + values[1] * self.MODEL_WEIGHTS[1]
                + values[2] * self.MODEL_WEIGHTS[2]
            )
            weighted_predictions[pathology] = float(weighted)
            confidence_intervals[pathology] = float(np.std(values))

            threshold = self.CLINICAL_THRESHOLDS.get(pathology, 0.50)
            above = [v >= threshold for v in values]
            if above.count(True) in {0, 3}:
                agreement_counter += 1
            total_counter += 1

        agreement_score = (agreement_counter / max(total_counter, 1)) * 100.0
        if resized_224 is None or tensor_224 is None:
            raise RuntimeError("NIH model output missing; cannot generate explainability images.")

        return (
            weighted_predictions,
            confidence_intervals,
            agreement_score,
            resized_224,
            tensor_224,
            model_outputs,
        )
