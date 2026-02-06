#!/usr/bin/env python3
"""
ONNX model evaluation with boundary-aware metrics.

Metrics computed:
- IoU (Intersection over Union)
- Dice Score
- BF Score (Boundary F1-score) - measures boundary alignment
- Boundary IoU - IoU computed only on boundary regions
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

import onnxruntime as ort
from scipy.ndimage import distance_transform_edt


def get_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """
    Extract boundary from a binary mask using morphological operations.

    Args:
        mask: Binary mask (H, W) with values 0 or 1
        dilation_ratio: Boundary thickness as ratio of image diagonal

    Returns:
        Binary boundary mask
    """
    h, w = mask.shape
    diag = np.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * diag)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Erode and dilate to get inner and outer boundaries
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=dilation)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation)

    # Boundary is the difference between dilated and eroded
    boundary = dilated - eroded
    return boundary.astype(np.float32)


def compute_bf_score(
    pred: np.ndarray,
    gt: np.ndarray,
    theta: float = 2.0
) -> float:
    """
    Compute Boundary F1-score (BF score).

    The BF score measures how well predicted boundaries align with ground truth
    boundaries, with tolerance controlled by theta (in pixels).

    Args:
        pred: Predicted binary mask (H, W)
        gt: Ground truth binary mask (H, W)
        theta: Distance tolerance in pixels (default: 2.0)

    Returns:
        BF score in [0, 1]
    """
    # Get boundaries
    pred_boundary = get_boundary(pred)
    gt_boundary = get_boundary(gt)

    # Get boundary pixels
    pred_boundary_pts = np.argwhere(pred_boundary > 0)
    gt_boundary_pts = np.argwhere(gt_boundary > 0)

    if len(pred_boundary_pts) == 0 and len(gt_boundary_pts) == 0:
        return 1.0  # Both empty, perfect match
    if len(pred_boundary_pts) == 0 or len(gt_boundary_pts) == 0:
        return 0.0  # One empty, no match

    # Compute distance transforms
    pred_dist = distance_transform_edt(1 - pred_boundary)
    gt_dist = distance_transform_edt(1 - gt_boundary)

    # Precision: fraction of predicted boundary within theta of gt boundary
    pred_in_gt = gt_dist[pred_boundary > 0]
    precision = np.mean(pred_in_gt <= theta) if len(pred_in_gt) > 0 else 0.0

    # Recall: fraction of gt boundary within theta of predicted boundary
    gt_in_pred = pred_dist[gt_boundary > 0]
    recall = np.mean(gt_in_pred <= theta) if len(gt_in_pred) > 0 else 0.0

    # F1 score
    if precision + recall == 0:
        return 0.0
    bf_score = 2 * precision * recall / (precision + recall)

    return bf_score


def compute_boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    dilation_ratio: float = 0.02
) -> float:
    """
    Compute Boundary IoU - IoU computed only on boundary regions.

    Args:
        pred: Predicted binary mask (H, W)
        gt: Ground truth binary mask (H, W)
        dilation_ratio: Boundary thickness as ratio of image diagonal

    Returns:
        Boundary IoU in [0, 1]
    """
    # Get boundaries
    pred_boundary = get_boundary(pred, dilation_ratio)
    gt_boundary = get_boundary(gt, dilation_ratio)

    # Create boundary region mask (union of both boundaries)
    boundary_region = ((pred_boundary > 0) | (gt_boundary > 0)).astype(np.float32)

    if boundary_region.sum() == 0:
        return 1.0 if (pred.sum() == 0 and gt.sum() == 0) else 0.0

    # Compute IoU only in boundary region
    pred_in_boundary = pred * boundary_region
    gt_in_boundary = gt * boundary_region

    intersection = (pred_in_boundary * gt_in_boundary).sum()
    union = pred_in_boundary.sum() + gt_in_boundary.sum() - intersection

    if union == 0:
        return 1.0
    return intersection / union


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union."""
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union == 0:
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return intersection / union


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score."""
    intersection = (pred * gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return 2 * intersection / total


def load_onnx_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model for inference."""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    return session


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def run_inference(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    """
    Run inference on a single image.

    Args:
        session: ONNX Runtime session
        image: Input image (H, W) normalized to [0, 1]

    Returns:
        Binary prediction mask (H, W)
    """
    # Prepare input
    input_name = session.get_inputs()[0].name
    x = image.astype(np.float32)

    # Add batch and channel dimensions if needed
    if x.ndim == 2:
        x = x[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    elif x.ndim == 3:
        x = x[np.newaxis, :, :, :]  # (1, C, H, W)

    # Run inference
    outputs = session.run(None, {input_name: x})
    logits = outputs[0]

    # Apply sigmoid and threshold
    probs = sigmoid(logits)
    pred = (probs > 0.5).astype(np.float32)

    # Remove batch and channel dimensions
    pred = pred.squeeze()
    return pred


def load_image(path: str) -> np.ndarray:
    """Load and preprocess image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Pad to 480x640 if needed
    h, w = img.shape
    if h < 480 or w < 640:
        pad_h = max(0, 480 - h)
        pad_w = max(0, 640 - w)
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    return img


def load_mask(path: str) -> np.ndarray:
    """Load ground truth mask."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")

    # Binarize
    mask = (mask > 127).astype(np.float32)

    # Pad to 480x640 if needed
    h, w = mask.shape
    if h < 480 or w < 640:
        pad_h = max(0, 480 - h)
        pad_w = max(0, 640 - w)
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    return mask


def evaluate_model(
    model_path: str,
    images_dir: str,
    masks_dir: str,
    theta: float = 2.0,
    dilation_ratio: float = 0.02
) -> Dict[str, float]:
    """
    Evaluate an ONNX model on a dataset.

    Args:
        model_path: Path to ONNX model
        images_dir: Directory containing test images
        masks_dir: Directory containing ground truth masks
        theta: BF score distance tolerance
        dilation_ratio: Boundary thickness ratio

    Returns:
        Dictionary with metric names and mean values
    """
    session = load_onnx_model(model_path)

    images_path = Path(images_dir)
    masks_path = Path(masks_dir)

    # Get all image files
    image_files = sorted(list(images_path.glob("*.png")) + list(images_path.glob("*.jpg")))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    metrics = {
        "iou": [],
        "dice": [],
        "bf_score": [],
        "boundary_iou": []
    }

    print(f"Evaluating {len(image_files)} images...")

    for img_file in image_files:
        # Find corresponding mask
        mask_file = masks_path / img_file.name
        if not mask_file.exists():
            # Try different extensions
            for ext in [".png", ".jpg", ".jpeg"]:
                alt_mask = masks_path / (img_file.stem + ext)
                if alt_mask.exists():
                    mask_file = alt_mask
                    break

        if not mask_file.exists():
            print(f"Warning: No mask found for {img_file.name}, skipping")
            continue

        # Load image and mask
        image = load_image(str(img_file))
        gt_mask = load_mask(str(mask_file))

        # Run inference
        pred_mask = run_inference(session, image)

        # Ensure same size
        h, w = gt_mask.shape
        pred_mask = pred_mask[:h, :w]

        # Compute metrics
        metrics["iou"].append(compute_iou(pred_mask, gt_mask))
        metrics["dice"].append(compute_dice(pred_mask, gt_mask))
        metrics["bf_score"].append(compute_bf_score(pred_mask, gt_mask, theta))
        metrics["boundary_iou"].append(compute_boundary_iou(pred_mask, gt_mask, dilation_ratio))

    # Compute means
    results = {k: np.mean(v) for k, v in metrics.items()}
    results["num_samples"] = len(metrics["iou"])

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX models with boundary-aware metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model or directory containing models")
    parser.add_argument("--images", type=str, default="data/images/test", help="Test images directory")
    parser.add_argument("--masks", type=str, default="data/masks/test", help="Test masks directory")
    parser.add_argument("--theta", type=float, default=2.0, help="BF score distance tolerance (pixels)")
    parser.add_argument("--dilation-ratio", type=float, default=0.02, help="Boundary thickness ratio")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file for results")
    args = parser.parse_args()

    model_path = Path(args.model)

    if model_path.is_file():
        # Single model
        models = [model_path]
    elif model_path.is_dir():
        # Directory of models - find all ONNX files
        models = sorted(model_path.rglob("*.onnx"))
    else:
        raise ValueError(f"Model path does not exist: {args.model}")

    all_results = []

    print("=" * 80)
    print("ONNX Model Evaluation with Boundary-Aware Metrics")
    print("=" * 80)
    print(f"Theta (BF tolerance): {args.theta} pixels")
    print(f"Boundary dilation ratio: {args.dilation_ratio}")
    print("=" * 80)

    for model_file in models:
        print(f"\nModel: {model_file}")
        print("-" * 60)

        try:
            results = evaluate_model(
                str(model_file),
                args.images,
                args.masks,
                args.theta,
                args.dilation_ratio
            )

            print(f"  Samples:      {results['num_samples']}")
            print(f"  IoU:          {results['iou']:.4f}")
            print(f"  Dice:         {results['dice']:.4f}")
            print(f"  BF Score:     {results['bf_score']:.4f}")
            print(f"  Boundary IoU: {results['boundary_iou']:.4f}")

            results["model"] = str(model_file)
            all_results.append(results)

        except Exception as e:
            print(f"  Error: {e}")

    # Save results to CSV if requested
    if args.output and all_results:
        import csv
        with open(args.output, "w", newline="") as f:
            fieldnames = ["model", "num_samples", "iou", "dice", "bf_score", "boundary_iou"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
