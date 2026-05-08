# Project 20: Synthetic-to-Real Edge Regularization

## Objective
A Staff-Level semantic segmentation pipeline that trains a U-Net with a **Total Variation (TV) denoising loss** to eliminate checkerboard and synthetic rendering artifacts on object boundaries. TV regularization penalizes abrupt inter-pixel transitions in the prediction probability map, producing visually smooth segmentation edges — particularly critical for the `Rocks` and `Trees` classes.

## Architecture Summary
| Component | Detail |
|---|---|
| Model | U-Net with **bilinear** upsampling (avoids transposed-conv checkerboards) |
| Loss | `CrossEntropy + λ × TotalVariation` |
| TV Weight (`λ`) | `0.001` (tunable via `--tv_weight` CLI flag) |
| Backbone | Scratch U-Net, 4 encoder levels |
| Optimizer | AdamW |

## Environment Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python tqdm scikit-learn
```

## Dataset Structure
```
data/
├── train/
│   ├── images/   (RGB .png/.jpg)
│   └── masks/    (Single-channel grayscale, class indices 0–9)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Training
```bash
# Default TV weight (0.001)
python train.py --epochs 50

# Higher TV weight for stronger edge smoothing
python train.py --epochs 50 --tv_weight 0.005
```
The training log separates `CE` and `TV` loss components each epoch so you can monitor
how much regularization is being applied relative to the main task.

## Inference & Edge Artifact Visualization
```bash
# Standard inference + metrics
python test.py

# Save high-contrast colorized side-by-side (GT | Prediction) PNGs for edge inspection
python test.py --save_vis
```
Visualizations are saved to `./visualizations/`. Open them and inspect the `Rocks` (silver)
and `Trees` (cyan) class boundaries for any remaining checkerboard or jagged artifacts.

## Tuning the TV Weight
| `tv_weight` | Effect |
|---|---|
| `0.0001` | Minimal smoothing; use as baseline |
| `0.001` | Recommended starting point |
| `0.005` | Strong smoothing; may slightly reduce mIoU |
| `0.01+` | Over-regularized; avoid unless artifacts are severe |

## Key Success Metrics
- **Visually smooth** prediction boundaries on `Rocks` and `Trees` (inspected via `--save_vis`)
- **Mean IoU ≥ 80%** on the clean test set
- **Inference time < 50 ms/image**
