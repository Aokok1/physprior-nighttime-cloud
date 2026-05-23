"""Global configuration for noise-label cloud detection project.

All paths and hyperparameters in one place. No hardcoding elsewhere.
"""
import os
from dataclasses import dataclass
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# MT-UNet data directory (shared, read-only). Actual data on this machine
# is at E:/Data/Unet_Dataset/ (not in the mtunet pipeline tree).
MTUNET_DATASET = "E:/Data/Unet_Dataset"

# Phase 3 dataset: removed 113 same-overpass + added 29 winter to Test
# Train: 2472 samples (was 2614), Test: 197 samples (was 168)
MTUNET_DATASET_PHASE3 = "E:/Data/Unet_Dataset"  # Uses Train_no_overlap subfolder
MTUNET_NORM_STATS = os.path.join(PROJECT_ROOT, "output", "norm_stats.npz")
MTUNET_CHECKPOINT = os.path.join(
    os.path.dirname(PROJECT_ROOT), "mtunet", "checkpoints", "best_mt_unet.pth"
)

# This project's output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ── Data ───────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    img_size: int = 128
    num_classes: int = 4
    # Class semantics: 0=True Clear, 1=prob_clear, 2=prob_cloud, 3=True Cloud
    # Binary cloud = classes 2 and 3
    binary_cloud_classes: tuple = (2, 3)

    # Clean label source: Center_Label field (radar-validated, 1 pixel per sample)
    # Noisy label source: Y_mask field (CLDMSK, full 128x128)


# ── Noise analysis ─────────────────────────────────────────────────────
@dataclass
class NoiseConfig:
    # Minimum radar samples needed per CLDMSK class to estimate noise rate
    min_samples_per_class: int = 20
    # Confidence threshold for SELFIE label replacement
    selfie_confidence_threshold: float = 0.85
    # Uncertainty threshold (for EDL teacher — if available)
    selfie_uncertainty_threshold: float = 0.3
    # SELFIE label replacement cap
    selfie_replace_ratio: float = 0.3


# ── Training ───────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 120
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    grad_clip: float = 5.0
    patience: int = 20
    num_workers: int = 0

    # Loss weights
    weight_center: float = 5.0      # radar anchor loss weight
    dice_weight: float = 0.5
    label_smoothing: float = 0.1

    # Co-teaching specific
    coteach_forget_rate: float = 0.4      # fraction of "noisy" samples to drop (matched to noise rate)
    coteach_num_gradual: int = 10         # epochs over which forget_rate ramps up

    # SELFIE specific
    selfie_teacher_mode: str = "standard"  # "standard" or "edl"
    selfie_replace_ratio: float = 0.3      # max fraction of pixels to relabel per sample

    # Cosine annealing
    T_0: int = 25
    T_mult: int = 2


# ── Model ──────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    backbone: str = "resnet34"
    in_channels: int = 3           # DNB + Basemap + M15
    mask_classes: int = 4
    use_film: bool = True
    use_evidential: bool = False   # True for EDL teacher in SELFIE


# ── Singleton instances ─────────────────────────────────────────────────
data_cfg = DataConfig()
noise_cfg = NoiseConfig()
train_cfg = TrainConfig()
model_cfg = ModelConfig()
