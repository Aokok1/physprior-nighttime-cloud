"""
建立统一数据源:frozen predictions。
为每个(method, test sample)生成 (pred, truth) 元组,把全表 5 个不同的
混淆矩阵全部 root out.
"""
import os, sys, json, hashlib
sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")

import numpy as np
import torch
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet
from datetime import date, timedelta

DEVICE = "cuda"
MTUNET_DATASET = r"E:/Data/Unet_Dataset"
CKPT_DIR = r"E:/Claude code/project/noise-label-cloud/output/checkpoints"

OUT_DIR = r"E:/Claude code/project/noise-label-cloud/output/frozen_predictions"
os.makedirs(OUT_DIR, exist_ok=True)


def _model_pred(model, ds, batch_size=8):
    """对每个 test sample 算 3x3 投票的二进制预测;存到 list of dicts。"""
    pred_records = []
    fname_set = set()
    for i in range(len(ds)):
        s = ds[i]
        cl = s["center_label"]
        if cl != cl:
            pred_records.append({"idx": i, "fname": s["filename"],
                                  "pred": None, "truth": None,
                                  "valid": False, "ry": int(s["radar_loc"][0]),
                                  "rx": int(s["radar_loc"][1])})
            fname_set.add(s["filename"])
            continue
        x = s["image"].unsqueeze(0).to(DEVICE)
        moon = torch.tensor([s["moon_phase"]], device=DEVICE)
        sza  = torch.tensor([s["solar_zenith"]], device=DEVICE)
        with torch.no_grad():
            logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
            pred_cls = logits.argmax(1)[0].cpu().numpy()
        ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
        y0, y1 = max(0, ry - 1), min(128, ry + 2)
        x0, x1 = max(0, rx - 1), min(128, rx + 2)
        win = pred_cls[y0:y1, x0:x1]
        bin_win = (win >= 2).astype(int)
        pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.mean() <= 0.5 else 1)
        pred_records.append({"idx": i, "fname": s["filename"],
                              "pred": int(pbin), "truth": int(cl),
                              "valid": True, "ry": ry, "rx": rx})
        fname_set.add(s["filename"])
    return pred_records


def _physprior_label_pred(ckpt_path=None, m15_min=264.0, std_max=1.5):
    """用 PhysPrior 校正后标签,在 radar 像素直接比较 — 这是 Ground Truth 转换。"""
    import glob
    from methods.physical_prior import apply_physical_correction
    pred_records = []
    files = sorted(glob.glob(os.path.join(MTUNET_DATASET, "Test", "*.npz")))
    for i, fp_ in enumerate(files):
        d = np.load(fp_)
        truth = d["Center_Label"]
        if truth != truth:
            pred_records.append({"idx": i, "fname": os.path.basename(fp_),
                                  "pred": None, "truth": None,
                                  "valid": False,
                                  "ry": int(d["Radar_Loc"][0]),
                                  "rx": int(d["Radar_Loc"][1])})
            continue
        cldmsk = d["Y_mask"].astype(np.int32)
        m15 = d["X_m15"]
        corrected, _, _ = apply_physical_correction(
            cldmsk, m15, m15_min=m15_min, std_max=std_max)
        ry, rx = int(d["Radar_Loc"][0]), int(d["Radar_Loc"][1])
        pred = 1 if corrected[ry, rx] in (2, 3) else 0
        pred_records.append({"idx": i, "fname": os.path.basename(fp_),
                              "pred": int(pred), "truth": int(truth),
                              "valid": True, "ry": ry, "rx": rx})
    return pred_records


def _season(fname):
    try:
        doy = int(fname.split("_A")[1].split(".")[0])
        year = doy // 1000
        ddd  = doy % 1000
        d = date(year, 1, 1) + timedelta(days=ddd - 1)
        m = d.month
    except Exception:
        return "Unknown"
    if m in (3,4,5): return "Spring"
    if m in (6,7,8): return "Summer"
    if m in (9,10,11): return "Autumn"
    return "Winter"


def _station(fname):
    if "Changsha" in fname: return "CS"
    if "Longmen" in fname: return "LM"
    return "?"


def attach_metadata(pred_records):
    """附加 season, station, overpass_yyyymmdd 字段。"""
    out = []
    for r in pred_records:
        d = dict(r)
        fname = d["fname"]
        d["season"] = _season(fname) if fname else "Unknown"
        d["station"] = _station(fname) if fname else "?"
        try:
            doy = int(fname.split("_A")[1].split(".")[0])
            d["overpass_yyyymmdd"] = doy
            d["overpass_date"] = (date(doy // 1000, 1, 1) +
                                   timedelta(days=doy % 1000 - 1)).isoformat()
        except Exception:
            d["overpass_yyyymmdd"] = None
            d["overpass_date"] = None
        out.append(d)
    return out


def _confusion(rs):
    tp = fp = tn = fn = 0
    for r in rs:
        if not r["valid"]: continue
        if   r["pred"]==1 and r["truth"]==1: tp += 1
        elif r["pred"]==1 and r["truth"]==0: fp += 1
        elif r["pred"]==0 and r["truth"]==0: tn += 1
        else: fn += 1
    return tp, fp, tn, fn


def _metrics(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    if n == 0: return None
    acc = (tp + tn) / n
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    sp = tn / (tn + fp) if tn + fp else 0
    f1 = 2 * p * r / max(p + r, 1e-9)
    # MCC
    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / denom
    # Balanced Accuracy
    ba = (r + sp) / 2
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "acc": acc, "precision": p, "recall": r, "f1": f1,
            "specificity": sp, "balanced_accuracy": ba, "mcc": mcc}


def _seeded_bootstrap_ci(corrects_arr, n_boot=2000, alpha=0.05, seed=42):
    np.random.seed(seed)
    n = len(corrects_arr)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        means[b] = corrects_arr[idx].mean()
    return float(np.percentile(means, alpha / 2 * 100)), \
           float(np.percentile(means, (1 - alpha / 2) * 100))


# ── 全部待评估的方法 ──
METHODS = {}

# 1. CLDMSK raw
print("Computing CLDMSK raw predictions...")
import glob
files = sorted(glob.glob(os.path.join(MTUNET_DATASET, "Test", "*.npz")))
raw_recs = []
for i, fp_ in enumerate(files):
    d = np.load(fp_)
    truth = d["Center_Label"]
    if truth != truth:
        raw_recs.append({"idx": i, "fname": os.path.basename(fp_),
                          "pred": None, "truth": None, "valid": False,
                          "ry": int(d["Radar_Loc"][0]), "rx": int(d["Radar_Loc"][1])})
        continue
    ry, rx = int(d["Radar_Loc"][0]), int(d["Radar_Loc"][1])
    c = int(d["Y_mask"][ry, rx])
    raw_recs.append({"idx": i, "fname": os.path.basename(fp_),
                      "pred": 1 if c in (2,3) else 0, "truth": int(truth),
                      "valid": True, "ry": ry, "rx": rx})
METHODS["CLDMSK_raw"] = attach_metadata(raw_recs)

# 2. Pure M15 thresholds (multiple)
print("Computing Pure M15 thresholds (T ∈ {260, 262, 264, 265, 266, 268})...")
for T in (260, 262, 264, 265, 266, 268):
    recs = []
    for i, fp_ in enumerate(files):
        d = np.load(fp_)
        truth = d["Center_Label"]
        if truth != truth:
            recs.append({"idx": i, "fname": os.path.basename(fp_),
                          "pred": None, "truth": None, "valid": False,
                          "ry": int(d["Radar_Loc"][0]), "rx": int(d["Radar_Loc"][1])})
            continue
        ry, rx = int(d["Radar_Loc"][0]), int(d["Radar_Loc"][1])
        c = int(d["Y_mask"][ry, rx])
        m15 = d["X_m15"][ry, rx]
        if np.isnan(m15): m15 = 270  # fallback
        # Pure M15 rule: CLDMSK says cloud → check if M15 warm
        if c in (2, 3) and m15 >= T:
            pred = 0
        else:
            pred = 1 if c in (2, 3) else 0
        recs.append({"idx": i, "fname": os.path.basename(fp_),
                      "pred": pred, "truth": int(truth), "valid": True,
                      "ry": ry, "rx": rx})
    METHODS[f"PureM15_T{T}"] = attach_metadata(recs)

# 3. PhysPrior label-only (no DL), with different presets
print("Computing PhysPrior label-only (moderate, conservative, aggressive)...")
for preset_name, T_th, sig_th in [
        ("moderate", 264.0, 1.5),
        ("conservative", 266.0, 1.0),
        ("aggressive", 262.0, 2.0)]:
    METHODS[f"PhysPriorLabel_{preset_name}"] = attach_metadata(
        _physprior_label_pred(m15_min=T_th, std_max=sig_th))

# 4. DL methods — load ckpts
ckpts = {
    "Baseline_phase4":       "phase4_baseline_best.pth",
    "PhysPrior_moderate":    "phase4_physprior_moderate_best.pth",
    "PhysPrior_ablation_no_basemap_2ch": "ablation_no_basemap_best.pth",
    "LossCorrection":        "loss_correction_best.pth",
    "CoTeaching_A":          "coteaching_a_best.pth",
    "CoTeaching_B":          "coteaching_b_best.pth",
    "GCE_q07":               "gce_q0.7_best.pth",
    "Mixup":                 "mixup_best.pth",
    "Baseline_origPhase_3ch": "baseline_best.pth",  # 含 leakage, 标记
    "PhysPrior_moderate_origPhase_3ch": "physprior_moderate_best.pth",  # 含 leakage, 标记
    "PhysPrior_conservative_orig": "physprior_conservative_best.pth",
    "PhysPrior_aggressive_orig":   "physprior_aggressive_best.pth",
    "Selfie_phys_teacher":   "selfie_phys_teacher.pth",
    "Selfie_student":        "selfie_student_best.pth",
    "PhysPrior_CoTeaching_long_a": "physprior_coteaching_long_a_best.pth",
    "PhysPrior_CoTeaching_long_b": "physprior_coteaching_long_b_best.pth",
    "PhysPrior_CoTeaching_a_short": "physprior_coteaching_a_best.pth",
    "PhysPrior_CoTeaching_b_short": "physprior_coteaching_b_best.pth",
}

# identify which need in_channels=2
need_in2 = {"PhysPrior_ablation_no_basemap_2ch"}

ds_3ch = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_basemap=True)
ds_2ch = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_basemap=False)

for method_name, ckpt_name in ckpts.items():
    cp = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.exists(cp):
        print(f"  (skipping {method_name}: {ckpt_name} not found)")
        continue
    in_channels = 2 if method_name in need_in2 else 3
    use_basemap = (in_channels == 3)
    ds = ds_2ch if in_channels == 2 else ds_3ch
    print(f"  Loading {method_name} from {ckpt_name}...")
    model = MT_UNet(backbone="resnet34", in_channels=in_channels, mask_classes=4,
                     use_film=True, use_evidential=False).to(DEVICE)
    sd = torch.load(cp, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    recs = _model_pred(model, ds)
    METHODS[method_name] = attach_metadata(recs)

# 5. Ensemble averaging for PhysPrior+Co-Teaching
print("Computing PhysPrior+CoTeaching ensemble (long run A+B averaged)...")
model_a = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                   use_film=True, use_evidential=False).to(DEVICE)
model_a.load_state_dict(torch.load(os.path.join(CKPT_DIR, "physprior_coteaching_long_a_best.pth"),
                                     map_location=DEVICE, weights_only=True))
model_a.eval()
model_b = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                   use_film=True, use_evidential=False).to(DEVICE)
model_b.load_state_dict(torch.load(os.path.join(CKPT_DIR, "physprior_coteaching_long_b_best.pth"),
                                     map_location=DEVICE, weights_only=True))
model_b.eval()
ens_recs = []
for i in range(len(ds_3ch)):
    s = ds_3ch[i]
    cl = s["center_label"]
    if cl != cl:
        ens_recs.append({"idx": i, "fname": s["filename"],
                          "pred": None, "truth": None, "valid": False,
                          "ry": int(s["radar_loc"][0]), "rx": int(s["radar_loc"][1])})
        continue
    x = s["image"].unsqueeze(0).to(DEVICE)
    moon = torch.tensor([s["moon_phase"]], device=DEVICE)
    sza  = torch.tensor([s["solar_zenith"]], device=DEVICE)
    with torch.no_grad():
        pa, _ = model_a(x, moon_phase=moon, solar_zenith=sza)
        pb, _ = model_b(x, moon_phase=moon, solar_zenith=sza)
        pred = (pa + pb) / 2
        pred_cls = pred.argmax(1)[0].cpu().numpy()
    ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
    y0, y1 = max(0, ry - 1), min(128, ry + 2)
    x0, x1 = max(0, rx - 1), min(128, rx + 2)
    win = pred_cls[y0:y1, x0:x1]
    bin_win = (win >= 2).astype(int)
    pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.mean() <= 0.5 else 1)
    ens_recs.append({"idx": i, "fname": s["filename"],
                     "pred": int(pbin), "truth": int(cl),
                     "valid": True, "ry": ry, "rx": rx})
METHODS["PhysPrior_CoTeaching_ensemble"] = attach_metadata(ens_recs)

# Save all
frozen_path = os.path.join(OUT_DIR, "all_frozen_predictions.json")
with open(frozen_path, "w") as f:
    json.dump(METHODS, f, indent=2)
print(f"\nFrozen predictions → {frozen_path}")
print(f"{len(METHODS)} methods frozen")
