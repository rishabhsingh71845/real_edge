import argparse
import time
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import logging

import config
from dataset import create_dataloaders
from model import UNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_metrics(conf_matrix: np.ndarray):
    """Calculate per-class IoU and Recall from a confusion matrix."""
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        recall = tp / (tp + fn)
    return iou, recall


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Map a single-channel class index mask to a high-contrast BGR color image.
    Uses the VIS_CLASS_COLORS palette defined in config.py.
    Useful for visually spotting edge artifacts on Rocks and Trees.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(config.VIS_CLASS_COLORS):
        color_mask[mask == cls_idx] = color
    return color_mask


def main():
    parser = argparse.ArgumentParser(
        description="Inference, metric evaluation, and edge artifact visualization"
    )
    parser.add_argument(
        "--save_vis", action="store_true",
        help="Save high-contrast colorized prediction maps to VIS_OUTPUT_DIR for edge inspection."
    )
    args = parser.parse_args()

    device = config.DEVICE
    logger.info(f"Running inference on device: {device}")

    _, test_loader = create_dataloaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,  # Dummy train dirs
        config.TEST_IMG_DIR, config.TEST_MASK_DIR,
        batch_size=1, num_workers=1,
    )

    if len(test_loader) == 0:
        logger.error("Test dataloader is empty. Populate 'data/test/' before running.")
        return

    model = UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(device)
    if os.path.exists(config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
        logger.info("Loaded best model checkpoint.")
    else:
        logger.warning("No checkpoint found — using randomly initialized weights.")

    model.eval()
    total_time = 0.0
    num_images = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for img_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Inference")):
            images = images.to(device)

            # --- Strict inference timing using perf_counter with GPU sync ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            logits = model(images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            inference_ms = (t_end - t_start) * 1000.0
            total_time += inference_ms
            num_images += 1

            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            true_mask = masks.squeeze(0).numpy()

            all_preds.extend(pred_mask.flatten())
            all_targets.extend(true_mask.flatten())

            # Optionally save high-contrast colorized prediction for visual inspection
            if args.save_vis:
                color_pred = colorize_mask(pred_mask)
                color_gt = colorize_mask(true_mask.astype(np.uint8))
                side_by_side = np.concatenate([color_gt, color_pred], axis=1)
                out_path = os.path.join(config.VIS_OUTPUT_DIR, f"vis_{img_idx:04d}.png")
                cv2.imwrite(out_path, side_by_side)

    avg_ms = total_time / num_images
    logger.info(f"Average inference time: {avg_ms:.2f} ms/image")
    if avg_ms > config.INFERENCE_SPEED_MS:
        logger.warning(
            f"Speed constraint FAILED: {avg_ms:.2f}ms > {config.INFERENCE_SPEED_MS}ms. "
            "Consider ONNX export or torch.compile() for acceleration."
        )
    else:
        logger.info(f"Speed constraint MET (< {config.INFERENCE_SPEED_MS}ms).")

    # Metrics
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(config.NUM_CLASSES)))
    ious, recalls = calculate_metrics(cm)
    mean_iou = float(np.nanmean(ious))

    logger.info(f"Mean IoU: {mean_iou * 100:.2f}%")
    logger.info(f"IoU — Rocks  (cls {config.ROCKS_CLASS_INDEX}): {ious[config.ROCKS_CLASS_INDEX] * 100:.2f}%")
    logger.info(f"IoU — Trees  (cls {config.TREES_CLASS_INDEX}): {ious[config.TREES_CLASS_INDEX] * 100:.2f}%")
    logger.info(f"Recall — Rocks: {recalls[config.ROCKS_CLASS_INDEX] * 100:.2f}%")
    logger.info(f"Recall — Trees: {recalls[config.TREES_CLASS_INDEX] * 100:.2f}%")

    if mean_iou * 100 >= config.TARGET_IOU:
        logger.info(f"Target IoU MET ({mean_iou * 100:.2f}% >= {config.TARGET_IOU}%).")
    else:
        logger.warning(f"Target IoU NOT MET ({mean_iou * 100:.2f}% < {config.TARGET_IOU}%).")

    if args.save_vis:
        logger.info(f"Visualizations saved to: {config.VIS_OUTPUT_DIR}")

    logger.info("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
