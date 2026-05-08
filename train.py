import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import create_dataloaders
from model import TotalVariationLoss, UNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_iou(preds, labels, num_classes):
    """Compute mean IoU across all classes."""
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        pred_mask = preds == cls
        true_mask = labels == cls
        intersection = (pred_mask & true_mask).sum().item()
        union = pred_mask.sum().item() + true_mask.sum().item() - intersection
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / max(union, 1))
    return float(np.nanmean(ious))


def train_epoch(model, loader, optimizer, ce_criterion, tv_criterion, tv_weight, scaler, device):
    model.train()
    total_loss = ce_total = tv_total = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            logits = model(images)
            ce_loss = ce_criterion(logits, masks)
            tv_loss = tv_criterion(logits)
            loss = ce_loss + tv_weight * tv_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        ce_total += ce_loss.item()
        tv_total += tv_loss.item()
    n = len(loader)
    return {"total": total_loss / n, "ce": ce_total / n, "tv": tv_total / n}


def val_epoch(model, loader, ce_criterion, device, num_classes):
    model.eval()
    epoch_loss = total_iou = batches = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = ce_criterion(logits, masks)
            epoch_loss += loss.item()
            total_iou += calculate_iou(logits, masks, num_classes)
            batches += 1
    return epoch_loss / len(loader), total_iou / max(batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net with TV Edge Regularization")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--tv_weight", type=float, default=config.TV_LOSS_WEIGHT)
    args = parser.parse_args()

    set_seed(config.SEED)
    logger.info(f"Project: {config.PROJECT_NAME}")
    logger.info(f"TV Loss weight: {args.tv_weight} | Device: {config.DEVICE}")

    train_loader, val_loader = create_dataloaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, config.VAL_MASK_DIR,
        config.BATCH_SIZE, config.NUM_WORKERS,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        logger.error("Dataloaders are empty. Populate 'data/' before training.")
        return

    device = config.DEVICE
    model = UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    tv_criterion = TotalVariationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.epochs} ---")
        train_losses = train_epoch(
            model, train_loader, optimizer, ce_criterion, tv_criterion, args.tv_weight, scaler, device
        )
        val_loss, val_iou = val_epoch(model, val_loader, ce_criterion, device, config.NUM_CLASSES)
        logger.info(
            f"Train Total={train_losses['total']:.4f} "
            f"(CE={train_losses['ce']:.4f}, TV={train_losses['tv']:.4f}) | "
            f"Val Loss={val_loss:.4f} | Val mIoU={val_iou:.4f}"
        )
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            logger.info(f"  New best model saved (mIoU={best_iou:.4f})")


if __name__ == "__main__":
    main()
