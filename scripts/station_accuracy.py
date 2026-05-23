"""
分站点计算 CLDMSK 夜间准确率
分别统计长沙(Changsha)和龙门(Longmen)的 CLDMSK vs 雷达 准确率
"""
import numpy as np
import os
import json
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
TRAIN_DIR = r"E:\Data\Unet_Dataset\Train"
TEST_DIR = r"E:\Data\Unet_Dataset\Test"
OUTPUT_DIR = r"E:\Claude code\project\noise-label-cloud\output"

# CLDMSK 二值化: 0,1=Clear(0), 2,3=Cloud(1)
CLDMSK_CLOUD_CLASSES = {2, 3}

def analyze_dir(data_dir, split_name):
    """分析一个目录下的分站点准确率"""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])
    print(f"\n{split_name}: {len(files)} files")
    
    station_data = defaultdict(lambda: {
        "total": 0, "has_radar": 0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "correct_list": [],  # for bootstrap
    })
    
    for fname in files:
        if "Changsha" in fname:
            station = "Changsha"
        elif "Longmen" in fname:
            station = "Longmen"
        else:
            continue
        
        station_data[station]["total"] += 1
        
        fpath = os.path.join(data_dir, fname)
        data = np.load(fpath, allow_pickle=True)
        
        # 获取中心像素
        y_mask = data["Y_mask"]
        cy, cx = y_mask.shape[0] // 2, y_mask.shape[1] // 2
        cldmsk_class = int(y_mask[cy, cx])
        
        # 雷达标签
        cl = data["Center_Label"]
        if cl.ndim == 0:
            radar_val = cl.item()
        else:
            radar_val = cl[cy, cx]
        
        if np.isnan(float(radar_val)):
            continue
        
        radar_binary = int(float(radar_val))  # 0=Clear, 1=Cloud
        cldmsk_binary = 1 if cldmsk_class in CLDMSK_CLOUD_CLASSES else 0
        
        station_data[station]["has_radar"] += 1
        
        # 混淆矩阵
        if cldmsk_binary == 1 and radar_binary == 1:
            station_data[station]["tp"] += 1
        elif cldmsk_binary == 1 and radar_binary == 0:
            station_data[station]["fp"] += 1
        elif cldmsk_binary == 0 and radar_binary == 1:
            station_data[station]["fn"] += 1
        elif cldmsk_binary == 0 and radar_binary == 0:
            station_data[station]["tn"] += 1
        
        station_data[station]["correct_list"].append(1 if cldmsk_binary == radar_binary else 0)
    
    # 输出
    results = {}
    for station in ["Changsha", "Longmen"]:
        d = station_data[station]
        n = d["has_radar"]
        if n == 0:
            print(f"\n  {station}: 无雷达数据")
            continue
        
        tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
        acc = (tp + tn) / n * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        
        # Bootstrap CI
        correct_arr = np.array(d["correct_list"])
        rng = np.random.RandomState(42)
        boot_accs = [rng.choice(correct_arr, size=n, replace=True).mean() * 100 for _ in range(2000)]
        ci_low = np.percentile(boot_accs, 2.5)
        ci_high = np.percentile(boot_accs, 97.5)
        
        print(f"\n  --- {station} (n={n}) ---")
        print(f"  总文件:     {d['total']}")
        print(f"  有雷达:     {n}")
        print(f"  Radar=Clear:{tn+fp}  Radar=Cloud:{tp+fn}")
        print(f"")
        print(f"  混淆矩阵:")
        print(f"                  Radar=Clear  Radar=Cloud")
        print(f"  CLDMSK=Clear    TN={tn:<4}       FN={fn:<4}")
        print(f"  CLDMSK=Cloud    FP={fp:<4}       TP={tp:<4}")
        print(f"")
        print(f"  准确率:   {acc:.2f}% [{ci_low:.2f}%, {ci_high:.2f}%]")
        print(f"  精确率:   {precision:.2f}%")
        print(f"  召回率:   {recall:.2f}%")
        print(f"  F1:       {f1:.2f}%")
        print(f"  误报率:   {far:.2f}%")
        
        results[station] = {
            "total_files": d["total"],
            "has_radar": n,
            "radar_clear": tn + fp,
            "radar_cloud": tp + fn,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": round(acc, 2),
            "ci_95": [round(ci_low, 2), round(ci_high, 2)],
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "far": round(far, 2),
        }
    
    # 合计
    all_tp = sum(station_data[s]["tp"] for s in ["Changsha", "Longmen"])
    all_fp = sum(station_data[s]["fp"] for s in ["Changsha", "Longmen"])
    all_fn = sum(station_data[s]["fn"] for s in ["Changsha", "Longmen"])
    all_tn = sum(station_data[s]["tn"] for s in ["Changsha", "Longmen"])
    all_n = all_tp + all_fp + all_fn + all_tn
    
    if all_n > 0:
        all_acc = (all_tp + all_tn) / all_n * 100
        all_prec = all_tp / (all_tp + all_fp) * 100 if (all_tp + all_fp) > 0 else 0
        all_rec = all_tp / (all_tp + all_fn) * 100 if (all_tp + all_fn) > 0 else 0
        all_f1 = 2 * all_prec * all_rec / (all_prec + all_rec) if (all_prec + all_rec) > 0 else 0
        all_far = all_fp / (all_fp + all_tn) * 100 if (all_fp + all_tn) > 0 else 0
        
        print(f"\n  --- 合计 (n={all_n}) ---")
        print(f"  混淆矩阵:")
        print(f"                  Radar=Clear  Radar=Cloud")
        print(f"  CLDMSK=Clear    TN={all_tn:<4}       FN={all_fn:<4}")
        print(f"  CLDMSK=Cloud    FP={all_fp:<4}       TP={all_tp:<4}")
        print(f"")
        print(f"  准确率:   {all_acc:.2f}%")
        print(f"  精确率:   {all_prec:.2f}%")
        print(f"  召回率:   {all_rec:.2f}%")
        print(f"  F1:       {all_f1:.2f}%")
        print(f"  误报率:   {all_far:.2f}%")
        
        results["combined"] = {
            "has_radar": all_n,
            "TP": all_tp, "FP": all_fp, "FN": all_fn, "TN": all_tn,
            "accuracy": round(all_acc, 2),
            "precision": round(all_prec, 2),
            "recall": round(all_rec, 2),
            "f1": round(all_f1, 2),
            "far": round(all_far, 2),
        }
    
    return results

def main():
    print("=" * 70)
    print("CLDMSK 夜间准确率 — 分站点统计")
    print("=" * 70)
    
    # Test set
    test_results = analyze_dir(TEST_DIR, "Test Set")
    
    # Train set
    train_results = analyze_dir(TRAIN_DIR, "Train Set")
    
    # 保存
    output = {"test": test_results, "train": train_results}
    out_path = os.path.join(OUTPUT_DIR, "cldmsk_station_accuracy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n\n结果已保存: {out_path}")

if __name__ == "__main__":
    main()
