"""Spearman correlation analysis: Daytime vs Nighttime CLDMSK.

Compares the correlation between CLDMSK accuracy and radar accuracy
for daytime vs nighttime samples. A key finding is that CLDMSK-IoU
is an inverse metric at night (higher IoU = lower radar accuracy).
"""
import os
import sys
import json
import numpy as np
from scipy import stats
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, OUTPUT_DIR


def load_samples_with_solar_zenith(data_dir):
    """Load samples and extract solar zenith angle (SZA) to determine day/night."""
    import glob
    
    samples = []
    for npz_path in glob.glob(os.path.join(data_dir, "*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        
        filename = os.path.basename(npz_path)
        sza = float(data.get("Solar_Zenith", 100.0))
        moon_phase = float(data.get("Moon_Phase", 180.0))
        center_label = float(data.get("Center_Label", float("nan")))
        
        # Get CLDMSK label at center pixel
        y_mask = np.squeeze(data["Y_mask"])
        center_y, center_x = 64, 64  # Center of 128x128
        cldmsk_class = int(y_mask[center_y, center_x])
        
        # Binary: 0=clear (classes 0,1), 1=cloud (classes 2,3)
        cldmsk_binary = 1 if cldmsk_class >= 2 else 0
        radar_binary = int(center_label) if not np.isnan(center_label) else None
        
        if radar_binary is not None:
            samples.append({
                "filename": filename,
                "solar_zenith": sza,
                "moon_phase": moon_phase,
                "cldmsk_binary": cldmsk_binary,
                "radar_binary": radar_binary,
                "cldmsk_class": cldmsk_class,
                "correct": int(cldmsk_binary == radar_binary),
            })
    
    return samples


def compute_spearman_by_sza(samples):
    """Compute Spearman correlation between SZA and CLDMSK accuracy."""
    sza_values = [s["solar_zenith"] for s in samples]
    accuracy_values = [s["correct"] for s in samples]
    
    # Spearman correlation
    rho, p_value = stats.spearmanr(sza_values, accuracy_values)
    
    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "n_samples": len(samples),
    }


def compute_accuracy_by_day_night(samples, sza_threshold=90.0):
    """Compute accuracy separately for daytime and nighttime."""
    daytime = [s for s in samples if s["solar_zenith"] < sza_threshold]
    nighttime = [s for s in samples if s["solar_zenith"] >= sza_threshold]
    
    results = {}
    
    for name, subset in [("daytime", daytime), ("nighttime", nighttime)]:
        if len(subset) > 0:
            acc = np.mean([s["correct"] for s in subset])
            
            # Bootstrap CI
            n_bootstrap = 2000
            boot_accs = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(subset), len(subset), replace=True)
                boot_acc = np.mean([subset[i]["correct"] for i in idx])
                boot_accs.append(boot_acc)
            
            ci_lower = np.percentile(boot_accs, 2.5)
            ci_upper = np.percentile(boot_accs, 97.5)
            
            results[name] = {
                "count": len(subset),
                "accuracy": float(acc),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "sza_range": [
                    float(min(s["solar_zenith"] for s in subset)),
                    float(max(s["solar_zenith"] for s in subset)),
                ],
            }
    
    return results


def compute_spearman_by_moon_phase(samples):
    """Compute Spearman correlation between moon phase and accuracy for nighttime."""
    nighttime = [s for s in samples if s["solar_zenith"] >= 90.0]
    
    if len(nighttime) < 10:
        return {"error": "Not enough nighttime samples"}
    
    moon_phases = [s["moon_phase"] for s in nighttime]
    accuracy_values = [s["correct"] for s in nighttime]
    
    rho, p_value = stats.spearmanr(moon_phases, accuracy_values)
    
    # Split by moon phase: bright (0-90, 270-360) vs dark (90-270)
    bright_moon = [s for s in nighttime if s["moon_phase"] < 90 or s["moon_phase"] > 270]
    dark_moon = [s for s in nighttime if 90 <= s["moon_phase"] <= 270]
    
    results = {
        "overall_rho": float(rho),
        "overall_p_value": float(p_value),
        "n_nighttime": len(nighttime),
    }
    
    for name, subset in [("bright_moon", bright_moon), ("dark_moon", dark_moon)]:
        if len(subset) > 0:
            acc = np.mean([s["correct"] for s in subset])
            results[name] = {
                "count": len(subset),
                "accuracy": float(acc),
            }
    
    return results


def main():
    """Run Spearman correlation analysis."""
    print("="*60)
    print("SPEARMAN CORRELATION ANALYSIS: DAYTIME vs NIGHTTIME")
    print("="*60)
    
    # Load samples
    print("\nLoading samples...")
    train_samples = load_samples_with_solar_zenith(os.path.join(MTUNET_DATASET, "Train"))
    test_samples = load_samples_with_solar_zenith(os.path.join(MTUNET_DATASET, "Test"))
    
    all_samples = train_samples + test_samples
    print(f"Total samples: {len(all_samples)} (Train: {len(train_samples)}, Test: {len(test_samples)})")
    
    # Overall Spearman correlation
    print("\n" + "-"*40)
    print("OVERALL SPEARMAN CORRELATION (SZA vs Accuracy)")
    print("-"*40)
    
    overall_spearman = compute_spearman_by_sza(all_samples)
    print(f"ρ = {overall_spearman['rho']:.3f} (p = {overall_spearman['p_value']:.4f})")
    print(f"n = {overall_spearman['n_samples']}")
    
    # Day vs Night accuracy
    print("\n" + "-"*40)
    print("DAYTIME vs NIGHTTIME CLDMSK ACCURACY")
    print("-"*40)
    
    day_night_results = compute_accuracy_by_day_night(all_samples)
    
    for name, stats in day_night_results.items():
        print(f"\n{name.upper()}:")
        print(f"  Count: {stats['count']}")
        print(f"  Accuracy: {stats['accuracy']:.1%} [{stats['ci_lower']:.1%}, {stats['ci_upper']:.1%}]")
        print(f"  SZA range: {stats['sza_range'][0]:.1f}° - {stats['sza_range'][1]:.1f}°")
    
    if "daytime" in day_night_results and "nighttime" in day_night_results:
        gap = day_night_results["daytime"]["accuracy"] - day_night_results["nighttime"]["accuracy"]
        print(f"\nDay-Night Gap: {gap:+.1%}")
    
    # Moon phase analysis
    print("\n" + "-"*40)
    print("MOON PHASE ANALYSIS (Nighttime Only)")
    print("-"*40)
    
    moon_results = compute_spearman_by_moon_phase(all_samples)
    
    if "error" not in moon_results:
        print(f"Moon Phase ρ = {moon_results['overall_rho']:.3f} (p = {moon_results['overall_p_value']:.4f})")
        print(f"Nighttime samples: {moon_results['n_nighttime']}")
        
        if "bright_moon" in moon_results:
            print(f"\nBright Moon: {moon_results['bright_moon']['count']} samples, "
                  f"accuracy = {moon_results['bright_moon']['accuracy']:.1%}")
        if "dark_moon" in moon_results:
            print(f"Dark Moon: {moon_results['dark_moon']['count']} samples, "
                  f"accuracy = {moon_results['dark_moon']['accuracy']:.1%}")
    else:
        print(moon_results["error"])
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "spearman_analysis.json")
    results = {
        "overall_spearman": overall_spearman,
        "day_night_accuracy": day_night_results,
        "moon_phase_analysis": moon_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if overall_spearman["rho"] > 0.3 and overall_spearman["p_value"] < 0.05:
        print("✓ Positive correlation: Higher SZA → higher accuracy")
        print("  This suggests CLDMSK performs better at night (counterintuitive!)")
    elif overall_spearman["rho"] < -0.3 and overall_spearman["p_value"] < 0.05:
        print("✗ Negative correlation: Higher SZA → lower accuracy")
        print("  This confirms CLDMSK performs worse at night")
    else:
        print("○ Weak or no correlation between SZA and accuracy")
    
    if "bright_moon" in moon_results and "dark_moon" in moon_results:
        bright_acc = moon_results["bright_moon"]["accuracy"]
        dark_acc = moon_results["dark_moon"]["accuracy"]
        if bright_acc > dark_acc + 0.05:
            print("\n✓ Moon phase matters: Bright moon → better accuracy")
            print("  DNB contributes to detection when moonlight is available")
        elif dark_acc > bright_acc + 0.05:
            print("\n? Dark moon → better accuracy (unexpected)")
            print("  May indicate other factors dominate")


if __name__ == "__main__":
    main()
