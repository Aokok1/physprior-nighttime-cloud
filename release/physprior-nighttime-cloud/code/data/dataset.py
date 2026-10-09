"""Dataset class for noise-label cloud detection.

Adapted from MT-UNet step4_unet.py. Reads .npz files and provides
both noisy (CLDMSK Y_mask) and clean (radar Center_Label) labels.
"""
import glob
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from config import data_cfg, MTUNET_NORM_STATS
from determinism import loader_generator

IMG_SIZE = data_cfg.img_size

# ── Basemap lookup ──────────────────────────────────────────────────────
# Every npz carries a baked-in X_basemap, which is a verbatim copy of one of
# the eight station-season composites in base_map/. Reading the composite at
# load time instead keeps that directory the single source of truth, so the
# basemap can be rebuilt without rewriting 2,790 npz files (they are hard-linked
# across the split directories, so an in-place rewrite would silently diverge).
# NL_BASEMAP_DIR selects the directory; if a composite is missing the baked-in
# copy is used, so the loader never fails for want of a basemap.
_BASEMAP_DIR = os.environ.get("NL_BASEMAP_DIR", "")
_basemap_cache = {}


def _season_from_tag(tag: str) -> str:
    """Julian-day season, matching scripts/rebuild_basemap.py."""
    doy = int(tag[5:8])
    if 60 <= doy <= 151:
        return "Spring"
    if 152 <= doy <= 243:
        return "Summer"
    if 244 <= doy <= 334:
        return "Autumn"
    return "Winter"


def _basemap_for(basename: str):
    """Return the station-season composite for a sample, or None."""
    if not _BASEMAP_DIR:
        return None
    parts = basename.split("_")
    if len(parts) < 3 or not parts[2].startswith("A"):
        return None
    site = parts[1].capitalize()
    season = _season_from_tag(parts[2])
    key = (site, season)
    if key not in _basemap_cache:
        p = os.path.join(_BASEMAP_DIR, f"Super_Basemap_{site}_{season}.npz")
        try:
            _basemap_cache[key] = np.load(p, allow_pickle=True)["X_dnb"]
        except Exception:
            _basemap_cache[key] = None
    return _basemap_cache[key]


# ── Normalization (shared with MT-UNet) ─────────────────────────────────
_stats = None


def _load_stats():
    global _stats
    if _stats is None:
        s = np.load(MTUNET_NORM_STATS)
        _stats = {
            "dnb_mean": float(s["dnb_mean"]),
            "dnb_std": float(s["dnb_std"]),
            "mod_mean": float(s["mod_mean"]),
            "mod_std": float(s["mod_std"]),
        }
    return _stats


def _log_norm(arr: np.ndarray) -> np.ndarray:
    s = _load_stats()
    safe = np.clip(arr, 1e-10, None)
    logv = np.log10(safe)
    return (logv - s["dnb_mean"]) / s["dnb_std"]


def _bt_norm(arr: np.ndarray) -> np.ndarray:
    s = _load_stats()
    result = (arr - s["mod_mean"]) / s["mod_std"]
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


# ── Dataset ─────────────────────────────────────────────────────────────
class CloudDataset(Dataset):
    """Loads .npz samples with both noisy (CLDMSK) and clean (radar) labels.

    Each sample contains:
      - image:  [C, 128, 128]  (DNB, [Basemap], M15)
      - mask_noisy: [128, 128]  CLDMSK 4-class (noisy label)
      - center_label: float     radar binary (clean, 1 pixel) or NaN
      - radar_loc: [2]          (ry, rx) of radar pixel
      - moon_phase, solar_zenith: float
    """

    def __init__(self, root_dir: str, mode: str = "Train", augment: bool = True,
                 use_basemap: bool = True, use_phase3: bool = False,
                 use_phase4: bool = False):
        self.mode = mode
        self.augment = augment and (mode == "Train")
        self.use_basemap = use_basemap
        # Phase 4: proper Train/Val split (fixes data leakage).
        # The suffix is selectable so that an alternative split can be evaluated
        # without touching any training script:
        #   NL_SPLIT_TAG=grp  ->  Train_grp / Val_grp  (group-disjoint, see
        #                         scripts/build_split_grp.py)
        #   unset             ->  Train_phase4 / Val_phase4  (original)
        split_tag = os.environ.get("NL_SPLIT_TAG", "phase4")
        if use_phase4:
            if mode == "Train":
                self.data_dir = os.path.join(root_dir, f"Train_{split_tag}")
            elif mode == "Val":
                self.data_dir = os.path.join(root_dir, f"Val_{split_tag}")
            else:
                self.data_dir = os.path.join(root_dir, mode)
        # Phase 3: use Train_no_overlap for training
        elif use_phase3 and mode == "Train":
            self.data_dir = os.path.join(root_dir, "Train_no_overlap")
        else:
            self.data_dir = os.path.join(root_dir, mode)
        self.file_list = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        if not self.file_list:
            raise ValueError(f"No .npz files found in {self.data_dir}")

    def __len__(self):
        return len(self.file_list)

    def _augment(self, x_norm, bm_norm, mod_norm, y_mask, y_height,
                 ry, rx):
        """Same augmentations as MT-UNet step4."""
        N = IMG_SIZE

        if random.random() > 0.5:
            sigma = random.uniform(0.02, 0.08)
            x_norm = x_norm + np.random.normal(0, sigma, x_norm.shape)
            mod_norm = mod_norm + np.random.normal(0, sigma * 0.5, mod_norm.shape)

        if random.random() > 0.5:
            scale = random.uniform(0.85, 1.15)
            x_norm = x_norm * scale
            bm_norm = bm_norm * scale

        if random.random() > 0.9:
            bm_norm = np.zeros_like(bm_norm)

        if random.random() > 0.5:
            x_norm = np.fliplr(x_norm)
            bm_norm = np.fliplr(bm_norm)
            mod_norm = np.fliplr(mod_norm)
            y_mask = np.fliplr(y_mask)
            y_height = np.fliplr(y_height)
            rx = N - 1 - rx

        if random.random() > 0.5:
            x_norm = np.flipud(x_norm)
            bm_norm = np.flipud(bm_norm)
            mod_norm = np.flipud(mod_norm)
            y_mask = np.flipud(y_mask)
            y_height = np.flipud(y_height)
            ry = N - 1 - ry

        k = random.randint(0, 3)
        for _ in range(k):
            x_norm = np.rot90(x_norm)
            bm_norm = np.rot90(bm_norm)
            mod_norm = np.rot90(mod_norm)
            y_mask = np.rot90(y_mask)
            y_height = np.rot90(y_height)
            ry, rx = rx, N - 1 - ry

        ry = int(np.clip(ry, 0, N - 1))
        rx = int(np.clip(rx, 0, N - 1))
        return x_norm, bm_norm, mod_norm, y_mask, y_height, ry, rx

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])

        x_dnb = data["X_dnb"]
        _bm = _basemap_for(os.path.basename(self.file_list[idx]))
        x_basemap = _bm if _bm is not None else data["X_basemap"]
        # Data has X_m15 (5ch version); fall back to X_mod (legacy 3ch)
        x_mod = data.get("X_m15", data.get("X_mod",
                 np.full((IMG_SIZE, IMG_SIZE), np.nan, dtype=np.float32)))
        y_mask = np.squeeze(data["Y_mask"]).copy()  # CLDMSK 4-class (NOISY); squeeze extra dims
        y_height = data["Y_height"].copy()

        radar_loc = data.get("Radar_Loc", np.array([IMG_SIZE // 2, IMG_SIZE // 2]))
        ry, rx = int(radar_loc[0]), int(radar_loc[1])

        center_label = float(data.get("Center_Label", float("nan")))  # radar CLEAN
        moon_phase = float(data.get("Moon_Phase", 180.0))
        solar_zenith = float(data.get("Solar_Zenith", 100.0))

        # Normalize
        x_norm = _log_norm(x_dnb)
        bm_norm = _log_norm(x_basemap)
        mod_norm = _bt_norm(x_mod)
        y_height_norm = y_height / 15000.0
        y_height_clean = np.nan_to_num(y_height_norm, nan=-1.0)

        # Augment
        if self.augment:
            x_norm, bm_norm, mod_norm, y_mask, y_height_clean, ry, rx = self._augment(
                x_norm, bm_norm, mod_norm, y_mask, y_height_clean, ry, rx
            )

        # Build tensors
        x_t = torch.from_numpy(x_norm.copy()).unsqueeze(0).float()
        mod_t = torch.from_numpy(mod_norm.copy()).unsqueeze(0).float()
        
        if self.use_basemap:
            bm_t = torch.from_numpy(bm_norm.copy()).unsqueeze(0).float()
            image = torch.cat([x_t, bm_t, mod_t], dim=0)
        else:
            image = torch.cat([x_t, mod_t], dim=0)

        return {
            "image": image,
            "mask_noisy": torch.from_numpy(y_mask.copy()).long(),
            "mask_noisy_soft": torch.from_numpy(y_mask.copy()).long(),
            "height": torch.from_numpy(y_height_clean.copy()).unsqueeze(0).float(),
            "center_label": center_label,
            "radar_loc": torch.tensor([ry, rx], dtype=torch.long),
            "moon_phase": moon_phase,
            "solar_zenith": solar_zenith,
            "filename": os.path.basename(self.file_list[idx]),
        }


def make_dataloaders(root_dir: str, batch_size: int = 16, num_workers: int = 0,
                     use_basemap: bool = True, use_phase3: bool = False,
                     use_phase4: bool = False):
    """Create Train/Val/Test dataloaders.
    
    Phase 4 (use_phase4=True): Proper Train/Val/Test split.
      - Train: Train_phase4/ (augmented, for gradient updates)
      - Val: Val_phase4/ (no augmentation, for early stopping/checkpoint)
      - Test: Test/ (never touched during training, for final eval only)
    """
    from torch.utils.data import DataLoader

    train_ds = CloudDataset(root_dir, "Train", augment=True, 
                            use_basemap=use_basemap, use_phase3=use_phase3,
                            use_phase4=use_phase4)
    val_ds = CloudDataset(root_dir, "Val", augment=False, 
                          use_basemap=use_basemap, use_phase3=use_phase3,
                          use_phase4=use_phase4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              generator=loader_generator(
                                  int(os.environ.get("NL_SEED", "42"))))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_loader, val_loader
