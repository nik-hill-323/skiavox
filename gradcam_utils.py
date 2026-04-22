"""Advanced explainability tools for Skiavox."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pytorch_grad_cam import EigenCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _to_rgb(img_array_2d: np.ndarray) -> np.ndarray:
    rgb = np.stack([img_array_2d, img_array_2d, img_array_2d], axis=-1).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


def generate_gradcam_plus_plus(model, img_tensor, img_array_2d, target_class_idx: int) -> Image.Image:
    """Generate GradCAM++ overlay for a target class."""
    cam = GradCAMPlusPlus(model=model, target_layers=[model.features.denseblock4])
    grayscale = cam(
        input_tensor=img_tensor, targets=[ClassifierOutputTarget(int(target_class_idx))]
    )[0, :]
    overlay = show_cam_on_image(_to_rgb(img_array_2d), grayscale, use_rgb=True)
    return Image.fromarray(overlay)


def generate_eigencam(model, img_tensor, img_array_2d) -> Image.Image:
    """Generate EigenCAM visualization."""
    cam = EigenCAM(model=model, target_layers=[model.features.denseblock4])
    grayscale = cam(input_tensor=img_tensor)[0, :]
    overlay = show_cam_on_image(_to_rgb(img_array_2d), grayscale, use_rgb=True)
    return Image.fromarray(overlay)


def annotate_regions(original_img: Image.Image, heatmap: np.ndarray, finding_name: str, confidence: float) -> Image.Image:
    """Annotate hottest region with a bounding box and label."""
    heat = np.asarray(heatmap, dtype=np.float32)
    threshold = np.percentile(heat, 98)
    ys, xs = np.where(heat >= threshold)
    annotated = original_img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        draw.rectangle([(x0, y0), (x1, y1)], outline=(239, 68, 68), width=3)
        draw.text(
            (x0 + 4, max(y0 - 20, 5)),
            f"{finding_name}: {confidence * 100:.1f}%",
            fill=(239, 68, 68),
        )
    return annotated


def generate_comparison_grid(model, img_tensor, img_array_2d, top_3_indices: Iterable[int]) -> Image.Image:
    """Create a 2x3 comparison grid with explainability views."""
    idxs: List[int] = list(top_3_indices)
    if len(idxs) < 3:
        idxs = idxs + [idxs[-1]] * (3 - len(idxs))

    rgb = _to_rgb(img_array_2d)
    cam_pp = GradCAMPlusPlus(model=model, target_layers=[model.features.denseblock4])
    eigen = EigenCAM(model=model, target_layers=[model.features.denseblock4])

    g1 = cam_pp(input_tensor=img_tensor, targets=[ClassifierOutputTarget(int(idxs[0]))])[0, :]
    g2 = cam_pp(input_tensor=img_tensor, targets=[ClassifierOutputTarget(int(idxs[1]))])[0, :]
    g3 = cam_pp(input_tensor=img_tensor, targets=[ClassifierOutputTarget(int(idxs[2]))])[0, :]
    e1 = eigen(input_tensor=img_tensor)[0, :]

    overlay1 = show_cam_on_image(rgb, g1, use_rgb=True)
    overlay2 = show_cam_on_image(rgb, g2, use_rgb=True)
    score_overlay = show_cam_on_image(rgb, (g1 + g2 + g3) / 3.0, use_rgb=True)
    ann = annotate_regions(Image.fromarray((rgb * 255).astype(np.uint8)), g1, "Top finding", 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].imshow(rgb, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(overlay1)
    axes[0, 1].set_title("GradCAM++ Finding 1")
    axes[0, 2].imshow(overlay2)
    axes[0, 2].set_title("GradCAM++ Finding 2")
    axes[1, 0].imshow(show_cam_on_image(rgb, e1, use_rgb=True))
    axes[1, 0].set_title("Eigen CAM")
    axes[1, 1].imshow(score_overlay)
    axes[1, 1].set_title("Score Overlay")
    axes[1, 2].imshow(ann)
    axes[1, 2].set_title("Annotated Regions")
    for ax in axes.flatten():
        ax.axis("off")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
