import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """
    Robust PyTorch Dataset for Semantic Segmentation.
    Handles corrupted or missing files gracefully by substituting a fallback sample.
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.valid_images = []
        if os.path.exists(image_dir) and os.path.exists(mask_dir):
            for img_name in os.listdir(image_dir):
                mask_path = os.path.join(mask_dir, img_name)
                if os.path.exists(mask_path):
                    self.valid_images.append(img_name)
                else:
                    logger.warning(f"Missing mask for image: {img_name}. Skipping.")
        else:
            logger.warning(
                f"Image or mask directory not found: '{image_dir}' / '{mask_dir}'. "
                "Populate with dataset before training."
            )

    def __len__(self) -> int:
        return len(self.valid_images)

    def __getitem__(self, index: int):
        img_name = self.valid_images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Mask is a single-channel grayscale map with integer class indices.
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                raise ValueError("File read returned None — likely corrupted.")
        except Exception as exc:
            logger.error(f"Failed to load '{img_name}': {exc}. Using fallback sample.")
            return self.__getitem__((index - 1) % max(1, len(self.valid_images)))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()


def get_train_transforms():
    """
    Standard geometric and color augmentations for training.
    No noise injection here; TV loss handles edge smoothness at the loss level.
    """
    return A.Compose(
        [
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms():
    """Clean validation transforms — no augmentation."""
    return A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def create_dataloaders(
    train_img: str,
    train_mask: str,
    val_img: str,
    val_mask: str,
    batch_size: int,
    num_workers: int,
):
    """Create and return train and validation DataLoaders."""
    train_ds = SegmentationDataset(train_img, train_mask, transform=get_train_transforms())
    val_ds = SegmentationDataset(val_img, val_mask, transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
