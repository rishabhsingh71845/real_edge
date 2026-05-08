import os
import torch

# General Project Configuration
PROJECT_NAME = "Synthetic_to_Real_Edge_Regularization"
SEED = 42

# Dataset Configuration
DATA_DIR = "./data"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "val", "masks")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test", "masks")

NUM_CLASSES = 10
CLASS_NAMES = [
    "Background", "Terrain", "Vegetation", "Sky", "Obstacle",
    "Rocks", "Water", "Vehicle", "Person", "Trees"
]
ROCKS_CLASS_INDEX = 5
TREES_CLASS_INDEX = 9

# Model Configuration
MODEL_BACKBONE = "unet"
IN_CHANNELS = 3
OUT_CHANNELS = NUM_CLASSES

# Training Configuration
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Total Variation (TV) Loss Weight
# Controls the trade-off between classification accuracy and edge smoothness.
# Higher TV_LOSS_WEIGHT = smoother boundaries, potentially lower mIoU.
TV_LOSS_WEIGHT = 0.001

# Hardware Toggles
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # Automatic Mixed Precision
NUM_WORKERS = 4

# Target Constraints
TARGET_IOU = 80.0
INFERENCE_SPEED_MS = 50.0

# Visualization: High-contrast color palette for edge artifact inspection (BGR format)
# One vibrant color per class to spot boundary artifacts.
VIS_CLASS_COLORS = [
    (0, 0, 0),       # Background: Black
    (128, 128, 0),   # Terrain: Olive
    (0, 200, 0),     # Vegetation: Green
    (255, 200, 0),   # Sky: Yellow
    (0, 0, 255),     # Obstacle: Blue
    (180, 180, 180), # Rocks: Silver
    (255, 0, 0),     # Water: Red
    (0, 165, 255),   # Vehicle: Orange
    (255, 0, 255),   # Person: Magenta
    (0, 255, 200),   # Trees: Cyan
]

# Paths
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
VIS_OUTPUT_DIR = "./visualizations"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
