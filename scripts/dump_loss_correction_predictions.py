"""Dump per-sample Loss Correction predictions from the saved re-estimated checkpoint.

The parent run (scripts/redo_loss_correction.py, 2026-07-14) reported only the four
confusion counts, so the row could not be bootstrapped or traced to per-sample
records. This script reloads output/checkpoints/loss_correction_best.pth, replays
the identical test-side protocol (3x3 logical-OR at the radar pixel) and writes the
per-sample records that the table builder consumes.

Run:  python scripts/dump_loss_correction_predictions.py
Out:  output/frozen_predictions/loss_correction_records.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")

from config import MTUNET_DATASET, model_cfg  # noqa: E402
from data.dataset import CloudDataset  # noqa: E402
from models.mt_unet import MT_UNet  # noqa: E402

CKPT = r"E:/Claude code/project/noise-label-cloud/output/checkpoints/loss_correction_best.pth"
OUT = r"E:/Claude code/project/noise-label-cloud/output/frozen_predictions/loss_correction_records.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

model = MT_UNet(
    backbone=model_cfg.backbone,
    in_channels=model_cfg.in_channels,
    mask_classes=model_cfg.mask_classes,
    use_film=model_cfg.use_film,
    use_evidential=False,
)
state = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
model = model.to(device).eval()

test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_basemap=True)
print(f"Test samples: {len(test_ds)}")

records = []
with torch.no_grad():
    for i in range(len(test_ds)):
        s = test_ds[i]
        cl = s["center_label"]
        if cl != cl:  # NaN -> no radar reference
            continue
        x = s["image"].unsqueeze(0).to(device)
        moon = torch.tensor([s["moon_phase"]], device=device)
        sza = torch.tensor([s["solar_zenith"]], device=device)
        logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = logits.argmax(1)[0].cpu().numpy()

        ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
        y0, y1 = max(0, ry - 1), min(128, ry + 2)
        x0, x1 = max(0, rx - 1), min(128, rx + 2)
        win = pred_cls[y0:y1, x0:x1]
        bw = (win >= 2).astype(int)
        pbin = 1 if bw.sum() > 0 else (0 if bw.mean() <= 0.5 else 1)

        records.append(
            {"fname": s["filename"], "pred": int(pbin), "truth": int(cl), "valid": True}
        )

arr = np.array([(r["pred"], r["truth"]) for r in records])
tp = int(((arr[:, 0] == 1) & (arr[:, 1] == 1)).sum())
fp = int(((arr[:, 0] == 1) & (arr[:, 1] == 0)).sum())
tn = int(((arr[:, 0] == 0) & (arr[:, 1] == 0)).sum())
fn = int(((arr[:, 0] == 0) & (arr[:, 1] == 1)).sum())
n = tp + fp + tn + fn
print(f"n={n}  TP={tp} FP={fp} TN={tn} FN={fn}  acc={(tp + tn) / n * 100:.2f}%")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(
        {
            "method": "Loss Correction (re-estimated transition matrix from train+val)",
            "checkpoint": "output/checkpoints/loss_correction_best.pth",
            "protocol": "3x3 logical-OR at radar pixel",
            "records": records,
        },
        f,
        indent=1,
    )
print(f"Saved -> {OUT}")
