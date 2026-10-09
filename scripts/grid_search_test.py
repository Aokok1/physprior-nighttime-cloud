"""P2 论文任务:48 网格搜索 on Test(确切证论文 Fig.1 / II-B "稳定区间 78-82%").
原来论文里没明确说网格搜索用 val 还是 test——这次在 test 上做"事后事实分析"。
"""
import os, sys, glob, json
import numpy as np
from scipy.ndimage import uniform_filter

TEST_DIR = r"E:/Data/Unet_Dataset/Test"

def _local_std(m15, win=5):
    m = m15.astype(np.float64)
    sqr_mean = uniform_filter(m, win)
    mean_sqr = uniform_filter(m ** 2, win)
    return np.sqrt(np.maximum(mean_sqr - sqr_mean ** 2, 0)).astype(np.float32)

def evaluate_grid(T_th, S_th, files):
    correct = 0; total = 0
    tp = fp = tn = fn = 0
    for fp_ in files:
        d = np.load(fp_)
        if d["Center_Label"] != d["Center_Label"]: continue
        m15 = d["X_m15"]
        if np.isnan(m15).all(): continue
        ry, rx = int(d["Radar_Loc"][0]), int(d["Radar_Loc"][1])
        std = _local_std(np.where(np.isnan(m15), np.nanmedian(m15), m15), 5)
        cldmsk_c = int(d["Y_mask"][ry, rx])
        local_std = std[ry, rx]
        m15_val = m15[ry, rx]
        if np.isnan(m15_val): continue
        # Apply PhysPrior-style threshold:
        if cldmsk_c in (2, 3) and m15_val >= T_th and local_std <= S_th:
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
    return {"acc": correct/max(total,1), "n": total,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}

def main():
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.npz")))
    print(f"Test files: {len(files)}, with radar labels:", end=" ")
    n_radar = sum(1 for fp_ in files if np.load(fp_)["Center_Label"] == np.load(fp_)["Center_Label"])
    print(n_radar)

    T_range = [258, 260, 262, 264, 266, 268, 270, 272]
    S_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    grid = {}
    matrix = np.zeros((len(T_range), len(S_range)))
    for ti, T in enumerate(T_range):
        for si, S in enumerate(S_range):
            r = evaluate_grid(T, S, files)
            key = f"T{T}_S{S}"
            grid[key] = r
            matrix[ti, si] = r["acc"] * 100

    print("\nGrid on TEST (n=197): accuracy (%)")
    print("T\\S", *[f"S={s:.1f}" for s in S_range], sep="\t")
    for ti, T in enumerate(T_range):
        row = [f"T={T}"]
        for si in range(len(S_range)):
            v = matrix[ti, si]
            mark = "*" if v == matrix.max() else " "
            row.append(f"{v:6.1f}{mark}")
        print(*row, sep="\t")

    best = max(grid.values(), key=lambda x: x["acc"])
    print(f"\nBest in test grid: {best['acc']*100:.1f}%  (TP={best['tp']} FP={best['fp']} TN={best['tn']} FN={best['fn']})")
    pure_T = evaluate_grid(266.0, 0.0, files)  # threshold w/o std filter
    pure_T_only = evaluate_grid(266.0, 999.0, files)  # equivalent: no std filter
    print(f"Pure M15 T=266K no std filter: {pure_T_only['acc']*100:.1f}%  TP={pure_T_only['tp']} FP={pure_T_only['fp']} TN={pure_T_only['tn']} FN={pure_T_only['fn']}")

    # Stable region analysis
    flat = matrix.flatten()
    print(f"\nMean: {flat.mean():.1f}%  Median: {np.median(flat):.1f}%  Std: {flat.std():.1f}%")
    print(f"% of grid ≥ 75%: {(flat >= 75).mean()*100:.0f}%  ≥ 70%: {(flat >= 70).mean()*100:.0f}%")

    out_path = r"E:/Claude code/project/noise-label-cloud/output/grid_test_postdoc.json"
    with open(out_path, "w") as f:
        json.dump({"matrix": matrix.tolist(),
                   "T_range": T_range, "S_range": S_range,
                   "best": best,
                   "grid": grid}, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
