"""Split Train_no_overlap into Train (80%) and Val (20%) with stratification.

This fixes the critical data leakage bug where Test set was used as validation.
After this split:
  - Train: for model training (with data augmentation)
  - Val: for early stopping / checkpoint selection (no augmentation)
  - Test: for final evaluation only (never seen during training)
"""
import os
import shutil
import random
import numpy as np
from collections import Counter, defaultdict

SEED = 42
VAL_RATIO = 0.20

BASE = "E:/Data/Unet_Dataset"
SRC_DIR = os.path.join(BASE, "Train_no_overlap")
TRAIN_DIR = os.path.join(BASE, "Train_phase4")
VAL_DIR = os.path.join(BASE, "Val_phase4")

random.seed(SEED)
np.random.seed(SEED)

# Collect all files
files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith(".npz")])
print(f"Source: {SRC_DIR}")
print(f"Total files: {len(files)}")

# Parse strata: (station, season)
def get_season(julian_day):
    month = (julian_day - 1) // 30 + 1
    if month > 12: month = 12
    if month in [12, 1, 2]: return "winter"
    elif month in [3, 4, 5]: return "spring"
    elif month in [6, 7, 8]: return "summer"
    else: return "autumn"

strata = defaultdict(list)
for f in files:
    # Tensor_Longmen_A2020286.1836.npz
    parts = f.replace("Tensor_", "").split("_A")
    station = parts[0]  # Changsha or Longmen
    year = int(parts[1][:4])
    julian = int(parts[1][4:7])
    season = get_season(julian)
    strata[(station, season)].append(f)

print(f"\nStrata distribution:")
for (st, ssn), fs in sorted(strata.items()):
    print(f"  {st:10s} {ssn:8s}: {len(fs):4d}")

# Stratified split
val_files = []
train_files = []
for (st, ssn), fs in strata.items():
    n_val = max(1, round(len(fs) * VAL_RATIO))
    shuffled = sorted(fs)  # deterministic order
    random.shuffle(shuffled)
    val_files.extend(shuffled[:n_val])
    train_files.extend(shuffled[n_val:])

print(f"\nSplit result:")
print(f"  Train: {len(train_files)} ({len(train_files)/len(files)*100:.1f}%)")
print(f"  Val:   {len(val_files)} ({len(val_files)/len(files)*100:.1f}%)")

# Verify no overlap
assert len(set(train_files) & set(val_files)) == 0, "Overlap detected!"
assert len(train_files) + len(val_files) == len(files), "Count mismatch!"

# Check radar coverage in each split
def count_radar(file_list, src_dir):
    has_radar = 0
    for f in file_list:
        d = np.load(os.path.join(src_dir, f), allow_pickle=True)
        rl = float(d["Center_Label"])
        if not np.isnan(rl):
            has_radar += 1
    return has_radar

print("\nRadar coverage check:")
train_radar = count_radar(train_files[:50], SRC_DIR)  # sample
val_radar = count_radar(val_files[:50], SRC_DIR)  # sample
print(f"  Train (sample 50): {train_radar}/50 have radar")
print(f"  Val (sample 50):   {val_radar}/50 have radar")

# Create directories and copy files
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

print(f"\nCopying {len(train_files)} files to {TRAIN_DIR}...")
for f in train_files:
    src = os.path.join(SRC_DIR, f)
    dst = os.path.join(TRAIN_DIR, f)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

print(f"Copying {len(val_files)} files to {VAL_DIR}...")
for f in val_files:
    src = os.path.join(SRC_DIR, f)
    dst = os.path.join(VAL_DIR, f)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

# Verify
train_actual = len([f for f in os.listdir(TRAIN_DIR) if f.endswith(".npz")])
val_actual = len([f for f in os.listdir(VAL_DIR) if f.endswith(".npz")])
print(f"\nVerification:")
print(f"  {TRAIN_DIR}: {train_actual} files")
print(f"  {VAL_DIR}: {val_actual} files")
print(f"  Total: {train_actual + val_actual} (expected {len(files)})")

# Save split manifest
manifest = {
    "seed": SEED,
    "val_ratio": VAL_RATIO,
    "total": len(files),
    "train_count": len(train_files),
    "val_count": len(val_files),
    "train_files": train_files,
    "val_files": val_files,
}
import json
manifest_path = os.path.join(BASE, "phase4_split_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest saved: {manifest_path}")
print("DONE")
