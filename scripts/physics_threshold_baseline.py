"""Pure physics threshold baseline — no deep learning.

Tests PhysPrior's M15 threshold + spatial std directly as a classifier,
without any neural network. This answers: "Does DL add value beyond
the physics rules?"

Method: 
  If M15 >= T_th AND spatial_std <= S_th → predict cloud
  Otherwise → predict clear

This is the simplest possible baseline that uses the same physics
as PhysPrior, but applies it directly to test data instead of
using it for label preprocessing.
"""
import os
import sys
import json
import numpy as np
from scipy.ndimage import uniform_filter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, OUTPUT_DIR


def compute_local_std(m15, window=5):
    """Compute local standard deviation of M15."""
    sqr_mean = uniform_filter(m15.astype(np.float64), window)
    mean_sqr = uniform_filter(m15.astype(np.float64) ** 2, window)
    var = mean_sqr - sqr_mean ** 2
    return np.sqrt(np.maximum(var, 0)).astype(np.float32)


def physics_threshold_predict(m15, t_th=264.0, s_th=1.5):
    """Predict cloud/clear using physics thresholds.
    
    Cloud if: M15 >= t_th AND std <= s_th (warm + uniform = likely clear sky)
    But CLDMSK says cloud → we flip to clear
    
    Actually, the logic is inverted for direct classification:
    - If M15 < t_th OR std > s_th → predict cloud (cold or textured)
    - If M15 >= t_th AND std <= s_th → predict clear (warm + uniform)
    """
    local_std = compute_local_std(m15, window=5)
    
    # Clear if warm AND uniform
    is_clear = (m15 >= t_th) & (local_std <= s_th)
    
    # Binary: 0=clear, 1=cloud
    prediction = (~is_clear).astype(int)
    return prediction


def evaluate_physics_threshold(split="Test", t_th=264.0, s_th=1.5):
    """Evaluate physics threshold on a dataset split."""
    data_dir = os.path.join(MTUNET_DATASET, split)
    
    correct = 0
    total = 0
    tp = fp = fn = tn = 0
    
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.npz'):
            continue
        
        filepath = os.path.join(data_dir, fname)
        data = np.load(filepath, allow_pickle=True)
        
        # Get radar label
        center_label = float(data.get("Center_Label", float("nan")))
        if np.isnan(center_label):
            continue
        
        # Get M15 (raw Kelvin, not normalized)
        m15 = data.get("X_m15", data.get("X_mod", None))
        if m15 is None:
            continue
        
        # Get radar location
        radar_loc = data.get("Radar_Loc", np.array([64, 64]))
        ry, rx = int(radar_loc[0]), int(radar_loc[1])
        
        # Predict using physics threshold
        pred = physics_threshold_predict(m15, t_th, s_th)
        
        # Get prediction at radar pixel (3x3 window)
        y0, y1 = max(0, ry-1), min(128, ry+2)
        x0, x1 = max(0, rx-1), min(128, rx+2)
        window = pred[y0:y1, x0:x1]
        
        # Cloud if any pixel in window is cloud
        pred_binary = 1 if window.sum() > 0 else 0
        true_binary = int(center_label)
        
        if pred_binary == true_binary:
            correct += 1
        total += 1
        
        # Confusion matrix
        if pred_binary == 1 and true_binary == 1:
            tp += 1
        elif pred_binary == 1 and true_binary == 0:
            fp += 1
        elif pred_binary == 0 and true_binary == 1:
            fn += 1
        else:
            tn += 1
    
    accuracy = correct / max(total, 1)
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "t_th": t_th,
        "s_th": s_th,
    }


def main():
    """Run physics threshold baseline with multiple thresholds."""
    print("="*60)
    print("Pure Physics Threshold Baseline (No DL)")
    print("="*60)
    
    # Test multiple threshold combinations
    thresholds = [
        (262.0, 2.0),  # aggressive
        (264.0, 1.5),  # moderate (PhysPrior default)
        (266.0, 1.0),  # conservative
        (260.0, 2.5),  # very aggressive
        (268.0, 0.5),  # very conservative
    ]
    
    results = []
    
    for t_th, s_th in thresholds:
        result = evaluate_physics_threshold("Test", t_th, s_th)
        results.append(result)
        print(f"\nT_th={t_th}K, S_th={s_th}K:")
        print(f"  Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
        print(f"  TP={result['tp']}, FP={result['fp']}, FN={result['fn']}, TN={result['tn']}")
    
    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n{'='*60}")
    print(f"Best: T_th={best['t_th']}K, S_th={best['s_th']}K")
    print(f"Accuracy: {best['accuracy']:.1%}")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "physics_threshold_baseline.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # Compare with PhysPrior
    print(f"\n{'='*60}")
    print("Comparison:")
    print(f"  Pure Physics Threshold: {best['accuracy']:.1%}")
    print(f"  PhysPrior + DL:         80.2%")
    print(f"  DL improvement:         {80.2 - best['accuracy']*100:+.1f}pp")


if __name__ == "__main__":
    main()
