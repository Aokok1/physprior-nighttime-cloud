"""复现论文 Table I-IV:载入已有 .pth ckpt,在 Test (n=197) 上评估。

测量的指标:
  - Layer 1 (主指标):雷达验证像素准确率 + 95% Bootstrap CI (2000 resamples)
  - 二元混淆矩阵 (TP/FP/TN/FN)
  - 纯 M15 阈值 (T=266K) baseline,无需训练
  - 站点分析 CS vs LM
  - 季节分析 Spring/Summer/Autumn/Winter
  - 通道消融
"""
import os, sys, json, glob, warnings
import numpy as np
import torch
from collections import defaultdict
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")

from config import MTUNET_DATASET
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = r"E:/Claude code/project/noise-label-cloud/output/checkpoints"
TEST_DIR = os.path.join(MTUNET_DATASET, "Test")


def _build_model(in_channels=3, use_film=True):
    return MT_UNet(
        backbone="resnet34", in_channels=in_channels, mask_classes=4,
        use_film=use_film, use_evidential=False,
    ).to(DEVICE)


def _load(ckpt_path, in_channels=3, use_film=True):
    model = _build_model(in_channels, use_film)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _bootstrap_ci(corrects, n_boot=2000, alpha=0.05):
    arr = np.array(corrects, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        means[b] = arr[idx].mean()
    return {
        "mean": float(arr.mean()),
        "lo":   float(np.percentile(means, alpha / 2 * 100)),
        "hi":   float(np.percentile(means, (1 - alpha / 2) * 100)),
        "n":    int(n),
    }


@torch.no_grad()
def predict_center(model, ds: CloudDataset, use_basemap=True):
    """对每个 Test 样本,用 3×3 窗口投票得到 (ry,rx) 处二元预测。"""
    model.eval()
    records = []
    for i in range(len(ds)):
        s = ds[i]
        x = s["image"].unsqueeze(0).to(DEVICE)
        moon = torch.tensor([s["moon_phase"]], device=DEVICE)
        sza  = torch.tensor([s["solar_zenith"]], device=DEVICE)
        ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
        logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = torch.argmax(logits, dim=1)[0].cpu().numpy()  # (128,128)
        y0, y1 = max(0, ry - 1), min(128, ry + 2)
        x0, x1 = max(0, rx - 1), min(128, rx + 2)
        win = pred_cls[y0:y1, x0:x1]
        binary_win = (win >= 2).astype(int)
        if binary_win.sum() > 0:
            pred_bin = 1
        elif binary_win.mean() <= 0.5:
            pred_bin = 0
        else:
            pred_bin = 1
        truth = s["center_label"]
        records.append({
            "fname": s["filename"],
            "pred": int(pred_bin),
            "truth": truth,
            "valid": not (truth != truth),  # non-NaN
            "ry": ry, "rx": rx,
        })
    return records


def _per_season(fname: str):
    """从文件名 (VIIRS ddddyyyy.hhmm) 推断季节(北半球)。"""
    # 例: Tensor_Changsha_A2020134.1824.npz
    try:
        doy = int(fname.split("_A")[1].split(".")[0])  # 2020134
        year = doy // 1000
        ddd = doy % 1000
        # 简化: 月=日序/30; 3-5春, 6-8夏, 9-11秋, 12-2冬
        month = min(12, max(1, int((ddd - 1) / 30.5) + 1))
        if month in (3, 4, 5): return "Spring"
        if month in (6, 7, 8): return "Summer"
        if month in (9, 10, 11): return "Autumn"
        return "Winter"
    except Exception:
        return "Unknown"


def _station(fname: str):
    if "Changsha" in fname: return "CS"
    if "Longmen" in fname: return "LM"
    return "?"


def _ci_from_tp_fp_tn_fn(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    if n == 0: return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    corrects = [1] * (tp + tn) + [0] * (fp + fn)
    arr = np.array(corrects, dtype=float)
    np.random.seed(0)
    means = np.empty(2000)
    for b in range(2000):
        idx = np.random.choice(len(arr), len(arr), replace=True)
        means[b] = arr[idx].mean()
    return {"mean": float(arr.mean()),
            "lo":   float(np.percentile(means, 2.5)),
            "hi":   float(np.percentile(means, 97.5))}


def evaluate_ckpt(name, ckpt_path, in_channels=3, use_film=True):
    print(f"\n>>> {name} ← {os.path.basename(ckpt_path)}")
    if not os.path.exists(ckpt_path):
        print("    (no checkpoint found)")
        return None

    model = _load(ckpt_path, in_channels, use_film)
    ds = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_basemap=(in_channels == 3))
    print(f"    Test set: {len(ds)} samples")

    recs = predict_center(model, ds)
    valid = [r for r in recs if r["valid"]]
    tp = sum(1 for r in valid if r["pred"] == 1 and r["truth"] == 1)
    fp = sum(1 for r in valid if r["pred"] == 1 and r["truth"] == 0)
    tn = sum(1 for r in valid if r["pred"] == 0 and r["truth"] == 0)
    fn = sum(1 for r in valid if r["pred"] == 0 and r["truth"] == 1)
    n = len(valid)
    acc = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    np.random.seed(42)
    ci = _ci_from_tp_fp_tn_fn(tp, fp, tn, fn)

    print(f"    Radar Acc = {acc:.3%} [95% CI {ci['lo']:.1%}, {ci['hi']:.1%}], n={n}")
    print(f"    Confusion  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"    P={precision:.1%}  R={recall:.1%}  F1={f1:.1%}")

    # Per-station
    station_buckets = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    for r in valid:
        s = _station(r["fname"])
        b = station_buckets[s]
        if r["pred"] == 1 and r["truth"] == 1: b["tp"] += 1
        elif r["pred"] == 1 and r["truth"] == 0: b["fp"] += 1
        elif r["pred"] == 0 and r["truth"] == 0: b["tn"] += 1
        else: b["fn"] += 1
    print("    Per-station:")
    for st, b in station_buckets.items():
        nn = b["tp"] + b["fp"] + b["tn"] + b["fn"]
        acc_s = (b["tp"] + b["tn"]) / max(nn, 1)
        p_s = b["tp"] / max(b["tp"] + b["fp"], 1)
        r_s = b["tp"] / max(b["tp"] + b["fn"], 1)
        f1_s = 2 * p_s * r_s / max(p_s + r_s, 1e-9)
        sp = b["tn"] / max(b["tn"] + b["fp"], 1)
        print(f"      {st}: n={nn} Acc={acc_s:.1%} P={p_s:.1%} R={r_s:.1%} F1={f1_s:.1%} Sp={sp:.1%}")

    # Seasonal
    season_buckets = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    for r in valid:
        ss = _per_season(r["fname"])
        b = season_buckets[ss]
        if r["pred"] == 1 and r["truth"] == 1: b["tp"] += 1
        elif r["pred"] == 1 and r["truth"] == 0: b["fp"] += 1
        elif r["pred"] == 0 and r["truth"] == 0: b["tn"] += 1
        else: b["fn"] += 1
    print("    Seasonal (DOY-based):")
    for sname in ("Spring", "Summer", "Autumn", "Winter"):
        b = season_buckets[sname]
        nn = b["tp"] + b["fp"] + b["tn"] + b["fn"]
        if nn == 0: continue
        acc_s = (b["tp"] + b["tn"]) / nn
        print(f"      {sname}: n={nn} Acc={acc_s:.1%}")

    # Month of test fnames overall
    return {
        "name": name, "n": n,
        "acc": acc, "ci": ci,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "station": {k: dict(v) for k, v in station_buckets.items()},
        "season":  {k: dict(v) for k, v in season_buckets.items()},
    }


# ── 纯 M15 阈值(论文中报道 77.2% @ T=266K)────────────────────────────────
def evaluate_pure_threshold(T_th=266.0):
    """无训练:对所有 Test 样本中心像素,CLDMSK -> 应用 M15 阈值后 -> 与雷达对比。"""
    print(f"\n>>> Pure M15 Threshold T={T_th}K (no training)")
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.npz")))
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    for fp_ in files:
        d = np.load(fp_)
        if d["Center_Label"] != d["Center_Label"]:  # NaN
            continue
        m15 = d["X_m15"]
        m15_f = np.where(np.isnan(m15), np.nanmedian(m15), m15)
        # Center pixel CLDMSK label
        ry, rx = int(d["Radar_Loc"][0]), int(d["Radar_Loc"][1])
        cldmsk_c = int(d["Y_mask"][ry, rx])
        # If CLDMSK says cloud but M15 >= T_th → flip to clear
        if cldmsk_c in (2, 3) and m15_f[ry, rx] >= T_th:
            pred = 0
        else:
            pred = 1 if cldmsk_c in (2, 3) else 0
        truth = int(d["Center_Label"])
        if pred == truth: correct += 1
        total += 1
        if   pred == 1 and truth == 1: tp += 1
        elif pred == 1 and truth == 0: fp += 1
        elif pred == 0 and truth == 0: tn += 1
        else: fn += 1
    if total == 0:
        print("    no valid samples"); return None
    acc = correct / total
    ci = _ci_from_tp_fp_tn_fn(tp, fp, tn, fn)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    print(f"    Radar Acc = {acc:.1%}  CI[{ci['lo']:.1%}, {ci['hi']:.1%}], n={total}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}  P={p:.1%} R={r:.1%} F1={f1:.1%}")
    return {"acc": acc, "ci": ci, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": p, "recall": r, "f1": f1, "n": total}


# ── 纯 CLDMSK baseline(无训练,直接用 CLDMSK 标签)───────────────────────────────
def evaluate_cldmsk_baseline():
    print(f"\n>>> Raw CLDMSK (no training)")
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.npz")))
    correct = 0; total = 0
    tp = fp = tn = fn = 0
    for fp_ in files:
        d = np.load(fp_)
        if d["Center_Label"] != d["Center_Label"]: continue
        ry, rx = int(d["Radar_Loc"][0]), int(d["Radar_Loc"][1])
        c = int(d["Y_mask"][ry, rx])
        pred = 1 if c in (2, 3) else 0
        truth = int(d["Center_Label"])
        if pred == truth: correct += 1
        total += 1
        if   pred == 1 and truth == 1: tp += 1
        elif pred == 1 and truth == 0: fp += 1
        elif pred == 0 and truth == 0: tn += 1
        else: fn += 1
    acc = correct / total
    ci = _ci_from_tp_fp_tn_fn(tp, fp, tn, fn)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    print(f"    Radar Acc = {acc:.1%}  CI[{ci['lo']:.1%}, {ci['hi']:.1%}], n={total}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}  P={p:.1%} R={r:.1%} F1={f1:.1%}")
    return {"acc": acc, "ci": ci, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": p, "recall": r, "f1": f1, "n": total}


if __name__ == "__main__":
    np.random.seed(42)
    results = {}

    # 0. No-training baselines
    results["CLDMSK_raw"] = evaluate_cldmsk_baseline()
    results["Pure_M15_T266K"] = evaluate_pure_threshold(266.0)
    results["Pure_M15_T264K"] = evaluate_pure_threshold(264.0)
    results["Pure_M15_T262K"] = evaluate_pure_threshold(262.0)

    # 1. Phase 4 ckpts (DATA LEAKAGE-FREE)
    results["phase4_baseline"] = evaluate_ckpt(
        "Phase4 Baseline (3ch, ckpt)",
        os.path.join(CKPT_DIR, "phase4_baseline_best.pth"),
        in_channels=3,
    )
    results["phase4_physprior_moderate"] = evaluate_ckpt(
        "Phase4 PhysPrior moderate (3ch, ckpt)",
        os.path.join(CKPT_DIR, "phase4_physprior_moderate_best.pth"),
        in_channels=3,
    )
    results["phase4_physprior_moderate_2ch"] = evaluate_ckpt(
        "Phase4 PhysPrior moderate (2ch no-basemap, ckpt)",
        os.path.join(CKPT_DIR, "ablation_no_basemap_best.pth"),
        in_channels=2,
    )

    # 2. Original-phase ckpts (可能有些 train-on-test 风险,仅参考)
    results["baseline_orig"] = evaluate_ckpt(
        "Original Baseline (3ch, ckpt)",
        os.path.join(CKPT_DIR, "baseline_best.pth"),
        in_channels=3,
    )
    results["loss_correction"] = evaluate_ckpt(
        "Loss Correction (3ch)",
        os.path.join(CKPT_DIR, "loss_correction_best.pth"),
        in_channels=3,
    )
    results["coteaching_a"] = evaluate_ckpt(
        "Co-Teaching Model A",
        os.path.join(CKPT_DIR, "coteaching_a_best.pth"),
        in_channels=3,
    )
    results["coteaching_b"] = evaluate_ckpt(
        "Co-Teaching Model B",
        os.path.join(CKPT_DIR, "coteaching_b_best.pth"),
        in_channels=3,
    )
    results["gce_q07"] = evaluate_ckpt(
        "GCE q=0.7",
        os.path.join(CKPT_DIR, "gce_q0.7_best.pth"),
        in_channels=3,
    )
    results["mixup"] = evaluate_ckpt(
        "Mixup",
        os.path.join(CKPT_DIR, "mixup_best.pth"),
        in_channels=3,
    )
    results["physprior_moderate_orig"] = evaluate_ckpt(
        "PhysPrior moderate (orig phase, ckpt)",
        os.path.join(CKPT_DIR, "physprior_moderate_best.pth"),
        in_channels=3,
    )
    results["physprior_aggressive"] = evaluate_ckpt(
        "PhysPrior aggressive",
        os.path.join(CKPT_DIR, "physprior_aggressive_best.pth"),
        in_channels=3,
    )
    results["physprior_conservative"] = evaluate_ckpt(
        "PhysPrior conservative",
        os.path.join(CKPT_DIR, "physprior_conservative_best.pth"),
        in_channels=3,
    )
    results["selfie_phys_teacher"] = evaluate_ckpt(
        "SELFIE PhysPrior teacher",
        os.path.join(CKPT_DIR, "selfie_phys_teacher.pth"),
        in_channels=3,
    )
    results["selfie_student"] = evaluate_ckpt(
        "SELFIE student",
        os.path.join(CKPT_DIR, "selfie_student_best.pth"),
        in_channels=3,
    )

    # ── 总结 ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("Table I (paper-style): Methods ranked by Radar Accuracy")
    print("=" * 70)
    rows = []
    for name, r in results.items():
        if not r:
            continue
        acc = r.get("acc")
        ci = r.get("ci", {})
        rows.append((name, acc, ci.get("lo", 0), ci.get("hi", 0)))
    rows.sort(key=lambda x: -x[1])
    for name, acc, lo, hi in rows:
        print(f"  {acc:>6.1%}  [{lo:.1%}, {hi:.1%}]   {name}")

    out_path = r"E:/Claude code/project/noise-label-cloud/output/verify_results.json"
    with open(out_path, "w") as f:
        json.dump({k: (v if isinstance(v, dict) else None) for k, v in results.items()},
                  f, indent=2, default=str)
    print(f"\nFull results → {out_path}")
