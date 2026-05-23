"""Seasonal analysis of noise-label cloud detection.

Extracts Julian day from filenames (e.g., Tensor_Changsha_A2019001.1824.npz)
and evaluates radar accuracy per season.

Seasons (Northern Hemisphere):
  Spring: March-May (Julian days 060-151)
  Summer: June-August (Julian days 152-243)
  Autumn: September-November (Julian days 244-334)
  Winter: December-February (Julian days 335-059)
"""
import os
import sys
import json
import re
import torch
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, CHECKPOINT_DIR, OUTPUT_DIR
from data import make_dataloaders
from models.mt_unet import MT_UNet
from eval import _collect_radar_predictions, _bootstrap_ci


def extract_julian_day(filename: str) -> int:
    """Extract Julian day from filename like Tensor_Changsha_A2019001.1824.npz"""
    match = re.search(r'A\d{4}(\d{3})\.', filename)
    if match:
        return int(match.group(1))
    return -1


def get_season(julian_day: int) -> str:
    """Map Julian day to season (Northern Hemisphere)."""
    if julian_day < 0:
        return "unknown"
    # Winter: Dec-Feb (335-365, 1-59)
    if julian_day >= 335 or julian_day < 60:
        return "winter"
    # Spring: Mar-May (60-151)
    elif julian_day < 152:
        return "spring"
    # Summer: Jun-Aug (152-243)
    elif julian_day < 244:
        return "summer"
    # Autumn: Sep-Nov (244-334)
    else:
        return "autumn"


@torch.no_grad()
def evaluate_by_season(model, dataloader, device="cuda"):
    """Evaluate radar accuracy stratified by season."""
    model.eval()
    records = _collect_radar_predictions(model, dataloader, device)
    
    # Add season info to records
    for rec in records:
        julian_day = extract_julian_day(rec["filename"])
        rec["julian_day"] = julian_day
        rec["season"] = get_season(julian_day)
    
    # Group by season
    by_season = defaultdict(list)
    for rec in records:
        by_season[rec["season"]].append(rec)
    
    # Compute stats per season
    results = {}
    for season, recs in sorted(by_season.items()):
        if len(recs) >= 5:
            results[season] = {
                "count": len(recs),
                "bootstrap_ci": _bootstrap_ci(recs),
                "julian_day_range": [
                    min(r["julian_day"] for r in recs),
                    max(r["julian_day"] for r in recs)
                ],
            }
    
    return results, records


def main():
    """Run seasonal analysis on best checkpoint."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: best in CHECKPOINT_DIR)")
    parser.add_argument("--method", type=str, default="physprior_moderate",
                        help="Method name for checkpoint lookup")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = MT_UNet(in_channels=3, mask_classes=4).to(device)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.method}_best.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"Available checkpoints:")
        for f in os.listdir(CHECKPOINT_DIR):
            if f.endswith(".pth"):
                print(f"  {f}")
        return
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")
    
    # Load data
    _, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=16)
    
    # Run seasonal analysis
    print("\n" + "="*60)
    print("SEASONAL ANALYSIS")
    print("="*60)
    
    results, records = evaluate_by_season(model, val_loader, device)
    
    # Print results
    print(f"\nTotal test samples: {len(records)}")
    print(f"\n{'Season':<10} {'Count':<8} {'Accuracy':<12} {'95% CI':<20}")
    print("-" * 50)
    
    for season, stats in sorted(results.items()):
        ci = stats["bootstrap_ci"]
        print(f"{season:<10} {stats['count']:<8} {ci['mean']:.1%} [{ci['ci_lower']:.1%}, {ci['ci_upper']:.1%}]")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, f"seasonal_analysis_{args.method}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Also save per-sample records for further analysis
    records_path = os.path.join(OUTPUT_DIR, f"seasonal_records_{args.method}.json")
    with open(records_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Per-sample records saved to: {records_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if "winter" in results and "summer" in results:
        winter_acc = results["winter"]["bootstrap_ci"]["mean"]
        summer_acc = results["summer"]["bootstrap_ci"]["mean"]
        diff = summer_acc - winter_acc
        print(f"Winter accuracy: {winter_acc:.1%}")
        print(f"Summer accuracy: {summer_acc:.1%}")
        print(f"Summer-Winter gap: {diff:+.1%}")
        print(f"\nNote: Winter typically has more temperature inversions,")
        print(f"which may cause more CLDMSK false positives (cold surface mimics cloud).")


if __name__ == "__main__":
    main()
