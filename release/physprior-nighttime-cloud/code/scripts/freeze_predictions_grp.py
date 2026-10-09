"""Freeze per-sample test predictions for the group-disjoint re-run.

Writes one record list per method into
output/frozen_predictions/all_frozen_predictions_grp.json, using the same
protocol as the published freeze step:
  - direct methods (CLDMSK, M15 thresholds, PhysPrior label rule): centre pixel
  - trained networks: 3x3 logical OR at the radar pixel

Checkpoints are read from the tag-aware CHECKPOINT_DIR, so
NL_CKPT_TAG=grp picks up output/checkpoints_grp.

Run:  NL_SPLIT_TAG=grp NL_CKPT_TAG=grp python scripts/freeze_predictions_grp.py
"""
import glob
import io
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import CHECKPOINT_DIR, MTUNET_DATASET  # noqa: E402
from data.dataset import CloudDataset  # noqa: E402
from models.mt_unet import MT_UNet  # noqa: E402
from methods.physical_prior import apply_physical_correction  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAG = os.environ.get("NL_CKPT_TAG", "")
SUF = f"_{TAG}" if TAG else ""
OUT = rf"E:/Claude code/project/noise-label-cloud/output/frozen_predictions/all_frozen_predictions{SUF}.json"

print(f"device={DEVICE}  CHECKPOINT_DIR={CHECKPOINT_DIR}  tag='{TAG}'")

test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_basemap=True)
print(f"Test samples: {len(test_ds)}")

METHODS = {}


def _valid(recs):
    return [r for r in recs if r.get("valid", True)]


# ---------------------------------------------------------------- direct rows
# Direct methods act on the raw label array and the raw M15 brightness
# temperature, so they are read straight from the .npz files rather than through
# CloudDataset (whose `image` tensor is normalised).
raw_recs = []
presets = {"conservative": (266.0, 1.0), "moderate": (264.0, 1.5), "aggressive": (262.0, 2.0)}
pp_by_preset = {k: [] for k in presets}
m15_by_T = {264.0: [], 266.0: []}

npz_files = sorted(glob.glob(os.path.join(MTUNET_DATASET, "Test", "*.npz")))
print(f"raw npz in Test: {len(npz_files)}")
for i, p in enumerate(npz_files):
    z = np.load(p, allow_pickle=True)
    fname = os.path.basename(p)
    cl = float(z["Center_Label"])
    if np.isnan(cl):
        rec = {"idx": i, "fname": fname, "pred": None, "truth": None, "valid": False}
        raw_recs.append(rec)
        for v in pp_by_preset.values():
            v.append(rec)
        for v in m15_by_T.values():
            v.append(rec)
        continue
    ym = np.asarray(z["Y_mask"]).astype(np.int32)
    m15 = np.asarray(z["X_m15"], dtype=np.float32)
    ry, rx = (int(x) for x in z["Radar_Loc"])
    truth = int(cl)
    center = int(ym[ry, rx])

    raw_recs.append({"idx": i, "fname": fname, "pred": 1 if center >= 2 else 0,
                     "truth": truth, "valid": True})

    for name, (T, S) in presets.items():
        corr, _n, _m = apply_physical_correction(ym, m15, m15_min=T, std_max=S)
        pp_by_preset[name].append({"idx": i, "fname": fname,
                                   "pred": 1 if int(corr[ry, rx]) >= 2 else 0,
                                   "truth": truth, "valid": True})
    for T in m15_by_T:
        # Pure-M15 rule, matching scripts/freeze_predictions.py: the same
        # one-directional flip as PhysPrior but with the texture condition
        # removed, so only the brightness-temperature test remains.
        mv = m15[ry, rx]
        if np.isnan(mv):
            mv = 270.0
        p = 0 if (center >= 2 and mv >= T) else (1 if center >= 2 else 0)
        m15_by_T[T].append({"idx": i, "fname": fname, "pred": int(p),
                            "truth": truth, "valid": True})

METHODS["CLDMSK_raw"] = raw_recs
for T, recs in m15_by_T.items():
    METHODS[f"PureM15_T{int(T)}"] = recs
for name, recs in pp_by_preset.items():
    METHODS[f"PhysPriorLabel_{name}"] = recs
print("direct rows done")

# ------------------------------------------------------------ trained networks
CKPTS = {
    "Standard_training": "baseline_best.pth",
    "LossCorrection": "loss_correction_best.pth",
    "CoTeaching_A": "coteaching_a_best.pth",
    "CoTeaching_B": "coteaching_b_best.pth",
    "GCE_q07": "gce_q0.7_best.pth",
    "Mixup": "mixup_best.pth",
    "PhysPrior_DL_3ch": "physprior_moderate_best.pth",
}


def eval_ckpt(path):
    model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                    use_film=True, use_evidential=False)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to(DEVICE).eval()
    recs = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            s = test_ds[i]
            cl = s["center_label"]
            if cl != cl:
                continue
            x = s["image"].unsqueeze(0).to(DEVICE)
            moon = torch.tensor([s["moon_phase"]], device=DEVICE)
            sza = torch.tensor([s["solar_zenith"]], device=DEVICE)
            logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
            pc = logits.argmax(1)[0].cpu().numpy()
            ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
            win = pc[max(0, ry - 1):ry + 2, max(0, rx - 1):rx + 2]
            bw = (win >= 2).astype(int)
            pbin = 1 if bw.sum() > 0 else 0
            recs.append({"idx": i, "fname": s["filename"], "pred": int(pbin),
                         "truth": int(cl), "valid": True})
    return recs


def eval_logits(path):
    """Per-sample full logit fields for one checkpoint.

    Only the ensemble row needs these: `train_physprior_coteaching_long.py`
    evaluates its two networks as `pred = (pa + pb) / 2` -- the two logit fields
    averaged -- and then applies the 3x3 OR. This function reproduces that,
    where `eval_ckpt` cannot, because a binary prediction has already thrown the
    logits away.
    """
    model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                    use_film=True, use_evidential=False)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            s = test_ds[i]
            cl = s["center_label"]
            if cl != cl:
                continue
            x = s["image"].unsqueeze(0).to(DEVICE)
            moon = torch.tensor([s["moon_phase"]], device=DEVICE)
            sza = torch.tensor([s["solar_zenith"]], device=DEVICE)
            logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
            out.append((s["filename"], int(cl),
                        int(s["radar_loc"][0]), int(s["radar_loc"][1]),
                        logits[0].cpu().numpy().astype(np.float32)))
    return out


for name, fn in CKPTS.items():
    p = os.path.join(CHECKPOINT_DIR, fn)
    if not os.path.exists(p):
        print(f"  SKIP {name}: {fn} not found")
        continue
    recs = eval_ckpt(p)
    METHODS[name] = recs
    acc = np.mean([r["pred"] == r["truth"] for r in recs])
    print(f"  {name:22s} n={len(recs)} acc={acc*100:.2f}%  ({fn})")

# ------------------------------------------------------------- 2x2 ablation
abl_p = rf"E:/Claude code/project/noise-label-cloud/output/2x2_ablation{SUF}.json"
if os.path.exists(abl_p):
    for cell in json.load(open(abl_p, encoding="utf-8")):
        key = f"abl_{cell['label_mode']}_{cell['n_channels']}ch"
        METHODS[key] = [{"fname": r["fname"], "pred": r["pred"], "truth": r["truth"], "valid": True}
                        for r in cell["test_records"]]
        print(f"  {key:22s} n={len(METHODS[key])}")
else:
    print(f"  SKIP 2x2 ablation: {abl_p} not found")

# ------------------------------------------------------------- five seeds
def _seed_entries(path, label):
    """Return a list of per-seed result dicts, tolerating a single-dict file."""
    if not os.path.exists(path):
        print(f"  SKIP {label}: {os.path.basename(path)} not found")
        return []
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        d = [d]
    out = [e for e in d if isinstance(e, dict) and "test_records" in e]
    if not out:
        print(f"  SKIP {label}: no per-sample records in {os.path.basename(path)}")
    return out


for label, path, prefix in (
    ("five-seed PhysPrior", rf"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_results{SUF}.json", "PhysPrior_5seed_seed"),
    ("five-seed Co-Teaching", rf"E:/Claude code/project/noise-label-cloud/output/coteaching_5seed_results{SUF}.json", "CoTeaching_5seed_seed"),
):
    for e in _seed_entries(path, label):
        key = f"{prefix}{e['seed']}"
        METHODS[key] = [{"fname": r["fname"], "pred": r["pred"], "truth": r["truth"], "valid": True}
                        for r in e["test_records"]]
        print(f"  {key:22s} n={len(METHODS[key])}")

# ------------------------------------------------------------- CoT ensemble
# The predefined ensemble is the one `train_physprior_coteaching_long.py`
# evaluates: the two networks' logit fields averaged, then argmax, then the
# 3x3 OR. An earlier version of this script instead OR-ed the two binary 3x3-OR
# results, which is a different ensemble under the same name (61.9% against
# 71.6% on the pre-rebuild basemaps).
ens = []
for label in ("a", "b"):
    p = os.path.join(CHECKPOINT_DIR, f"physprior_coteaching_long_{label}_best.pth")
    if os.path.exists(p):
        ens.append(eval_logits(p))
if len(ens) == 2:
    both = []
    for ra, rb in zip(ens[0], ens[1]):
        if ra[0] != rb[0]:
            raise SystemExit(f"ensemble sample order mismatch: {ra[0]} vs {rb[0]}")
        pred_cls = ((ra[4] + rb[4]) / 2.0).argmax(0)
        ry, rx = ra[2], ra[3]
        win = pred_cls[max(0, ry - 1):ry + 2, max(0, rx - 1):rx + 2]
        bw = (win >= 2).astype(int)
        pbin = 1 if bw.sum() > 0 else (0 if bw.mean() <= 0.5 else 1)
        both.append({"fname": ra[0], "pred": int(pbin), "truth": ra[1], "valid": True})
    METHODS["PhysPrior_CoTeaching_ensemble"] = both
    acc = float(np.mean([r["pred"] == r["truth"] for r in both]))
    print(f"  CoTeaching ensemble     n={len(both)} acc={acc*100:.2f}%  (logit average)")
else:
    print("  SKIP CoT ensemble: checkpoints not found")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(METHODS, f, indent=1)
print(f"\nwrote {OUT}  ({len(METHODS)} methods)")
