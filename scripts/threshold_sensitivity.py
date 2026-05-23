"""Threshold sensitivity analysis for Physical Prior Correction.

Sweeps M15 temperature threshold and spatial std threshold to show
robustness of the physical prior approach. Evaluates on 168 radar
validation pixels without retraining — pure label-level correction.

Usage:
    python scripts/threshold_sensitivity.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from collections import defaultdict

from config import MTUNET_DATASET


def load_radar_pixels(root_dir):
    """Load all test samples with radar labels."""
    import glob
    test_dir = os.path.join(root_dir, "Test")
    files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))

    samples = []
    for f in files:
        data = np.load(f)
        center = float(data.get("Center_Label", float("nan")))
        if np.isnan(center):
            continue
        x_m15 = data.get("X_m15", data.get("X_mod", None))
        if x_m15 is None:
            continue
        radar_loc = data.get("Radar_Loc", np.array([64, 64]))
        ry, rx = int(radar_loc[0]), int(radar_loc[1])
        y_mask = np.squeeze(data["Y_mask"]).copy()

        # Extract M15 at radar pixel (3x3 window)
        H, W = x_m15.shape
        y0, y1 = max(0, ry - 1), min(H, ry + 2)
        x0, x1 = max(0, rx - 1), min(W, rx + 2)
        m15_win = x_m15[y0:y1, x0:x1]
        m15_mean = float(np.nanmean(m15_win))
        m15_std = float(np.nanstd(m15_win))

        # CLDMSK label at radar pixel
        cldmsk_label = int(y_mask[ry, rx])
        # Binary: 0=clear (classes 0,1), 1=cloud (classes 2,3)
        cldmsk_binary = 1 if cldmsk_label >= 2 else 0
        radar_binary = int(center)

        samples.append({
            "m15_mean": m15_mean,
            "m15_std": m15_std,
            "cldmsk_binary": cldmsk_binary,
            "radar_binary": radar_binary,
            "cldmsk_class": cldmsk_label,
        })

    return samples


def apply_phys_prior(samples, t_min, std_max):
    """Apply physical prior correction and compute accuracy."""
    correct = 0
    total = len(samples)
    tp = fp = fn = tn = 0

    for s in samples:
        # Physical prior rule: warm + uniform → flip to clear
        if s["m15_mean"] >= t_min and s["m15_std"] <= std_max:
            pred = 0  # flip to clear
        else:
            pred = s["cldmsk_binary"]  # keep original

        if pred == s["radar_binary"]:
            correct += 1
        if pred == 1 and s["radar_binary"] == 1:
            tp += 1
        elif pred == 1 and s["radar_binary"] == 0:
            fp += 1
        elif pred == 0 and s["radar_binary"] == 1:
            fn += 1
        else:
            tn += 1

    acc = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "correct": correct, "total": total,
    }


def bootstrap_ci(samples, t_min, std_max, n_boot=2000):
    """Bootstrap 95% CI for accuracy."""
    n = len(samples)
    accs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot = [samples[i] for i in idx]
        result = apply_phys_prior(boot, t_min, std_max)
        accs.append(result["accuracy"])
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def main():
    print("Loading radar validation pixels...")
    samples = load_radar_pixels(MTUNET_DATASET)
    print(f"  Loaded {len(samples)} radar-validated test pixels\n")

    # Raw CLDMSK accuracy
    raw_correct = sum(1 for s in samples if s["cldmsk_binary"] == s["radar_binary"])
    raw_acc = raw_correct / len(samples)
    print(f"Raw CLDMSK accuracy: {raw_acc:.2%} ({raw_correct}/{len(samples)})\n")

    # ── Sweep M15 threshold (fix std=1.5K) ──
    print("=" * 70)
    print("  M15 Temperature Threshold Sweep (std_max = 1.5K fixed)")
    print("=" * 70)
    print(f"{'T_min (K)':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'FP_red':>8} {'TP_loss':>8}")
    print("-" * 70)

    raw_fp = sum(1 for s in samples if s["cldmsk_binary"] == 1 and s["radar_binary"] == 0)
    raw_tp = sum(1 for s in samples if s["cldmsk_binary"] == 1 and s["radar_binary"] == 1)

    for t_min in [258, 260, 262, 264, 266, 268, 270, 272]:
        r = apply_phys_prior(samples, t_min, 1.5)
        new_fp = r["fp"]
        fp_reduction = (raw_fp - new_fp) / raw_fp * 100 if raw_fp > 0 else 0
        new_tp = r["tp"]
        tp_loss = (raw_tp - new_tp) / raw_tp * 100 if raw_tp > 0 else 0
        print(f"{t_min:>10.0f} {r['accuracy']:>10.2%} {r['precision']:>10.2%} "
              f"{r['recall']:>10.2%} {r['f1']:>8.3f} {fp_reduction:>7.1f}% {tp_loss:>7.1f}%")

    # ── Sweep std threshold (fix T=264K) ──
    print(f"\n{'='*70}")
    print("  Spatial Std Threshold Sweep (T_min = 264K fixed)")
    print("=" * 70)
    print(f"{'std_max (K)':>12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'FP_red':>8} {'TP_loss':>8}")
    print("-" * 70)

    for std_max in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        r = apply_phys_prior(samples, 264, std_max)
        new_fp = r["fp"]
        fp_reduction = (raw_fp - new_fp) / raw_fp * 100 if raw_fp > 0 else 0
        new_tp = r["tp"]
        tp_loss = (raw_tp - new_tp) / raw_tp * 100 if raw_tp > 0 else 0
        print(f"{std_max:>12.1f} {r['accuracy']:>10.2%} {r['precision']:>10.2%} "
              f"{r['recall']:>10.2%} {r['f1']:>8.3f} {fp_reduction:>7.1f}% {tp_loss:>7.1f}%")

    # ── 2D grid search ──
    print(f"\n{'='*70}")
    print("  2D Grid Search: T_min × std_max → Radar Accuracy")
    print("=" * 70)

    t_range = [258, 260, 262, 264, 266, 268, 270, 272]
    s_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Header
    header = "T\\std"
    print(f"{header:>8}", end="")
    for s in s_range:
        print(f"{s:>8.1f}", end="")
    print()
    print("-" * (8 + 8 * len(s_range)))

    best_acc = 0
    best_params = (0, 0)
    grid_results = {}

    for t in t_range:
        print(f"{t:>8.0f}", end="")
        for s in s_range:
            r = apply_phys_prior(samples, t, s)
            acc = r["accuracy"]
            grid_results[f"{t}_{s}"] = r
            print(f"{acc:>8.2%}", end="")
            if acc > best_acc:
                best_acc = acc
                best_params = (t, s)
        print()

    print(f"\nBest: T_min={best_params[0]}K, std_max={best_params[1]}K → {best_acc:.2%}")

    # ── Bootstrap CI for best params ──
    print(f"\nBootstrap 95% CI for best params (T={best_params[0]}, std={best_params[1]})...")
    ci_lo, ci_hi = bootstrap_ci(samples, best_params[0], best_params[1])
    print(f"  95% CI: [{ci_lo:.2%}, {ci_hi:.2%}]")

    # ── Save results ──
    output = {
        "raw_cldmsk_accuracy": raw_acc,
        "n_samples": len(samples),
        "best_params": {"t_min": best_params[0], "std_max": best_params[1]},
        "best_accuracy": best_acc,
        "best_ci_95": [float(ci_lo), float(ci_hi)],
        "grid_results": {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv
                           for kk, vv in v.items()} for k, v in grid_results.items()},
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "output", "threshold_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
