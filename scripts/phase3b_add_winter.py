"""Phase 3b: Move winter samples from Train to Test for seasonal coverage.

Current Test set has NO winter samples. This script:
1. Identifies winter samples in Train that have radar labels
2. Moves them to Test set
3. Reports new seasonal distribution
"""
import os
import sys
import re
import shutil
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET


def get_season(julian_day):
    """Map Julian day to season."""
    if julian_day >= 335 or julian_day < 60:
        return "winter"
    elif julian_day < 152:
        return "spring"
    elif julian_day < 244:
        return "summer"
    else:
        return "autumn"


def extract_julian_day(filename):
    """Extract Julian day from filename."""
    match = re.search(r'A\d{4}(\d{3})\.', filename)
    if match:
        return int(match.group(1))
    return -1


def has_radar_label(filepath):
    """Check if sample has valid radar label."""
    try:
        data = np.load(filepath)
        cl = float(data.get("Center_Label", float("nan")))
        return not np.isnan(cl)
    except:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be moved")
    parser.add_argument("--max-move", type=int, default=50,
                        help="Max winter samples to move to test")
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 3b: Add Winter Samples to Test Set")
    print("="*60)
    
    train_dir = os.path.join(MTUNET_DATASET, "Train")
    test_dir = os.path.join(MTUNET_DATASET, "Test")
    
    # Find winter samples in train with radar labels
    winter_with_radar = []
    winter_without_radar = []
    
    for f in os.listdir(train_dir):
        if not f.endswith('.npz'):
            continue
        jd = extract_julian_day(f)
        season = get_season(jd)
        if season == "winter":
            filepath = os.path.join(train_dir, f)
            if has_radar_label(filepath):
                winter_with_radar.append(f)
            else:
                winter_without_radar.append(f)
    
    print(f"\nWinter samples in Train:")
    print(f"  With radar: {len(winter_with_radar)}")
    print(f"  Without radar: {len(winter_without_radar)}")
    
    # Current seasonal distribution
    for split, dir_path in [("Train", train_dir), ("Test", test_dir)]:
        seasons = {"winter": 0, "spring": 0, "summer": 0, "autumn": 0}
        for f in os.listdir(dir_path):
            if f.endswith('.npz'):
                jd = extract_julian_day(f)
                seasons[get_season(jd)] += 1
        print(f"\n{split} seasonal distribution:")
        for s, count in seasons.items():
            print(f"  {s}: {count}")
    
    # Select winter samples to move (prefer those with radar)
    to_move = winter_with_radar[:args.max_move]
    
    if not to_move:
        print("\nNo winter samples with radar labels found!")
        print("Moving winter samples without radar labels instead...")
        to_move = winter_without_radar[:args.max_move]
    
    print(f"\nWill move {len(to_move)} winter samples to Test")
    
    if args.dry_run:
        print("\n[DRY RUN] Sample files that would be moved:")
        for f in to_move[:10]:
            print(f"  {f}")
        print(f"  ... and {max(0, len(to_move)-10)} more")
        return
    
    # Create backup of test dir
    test_backup = test_dir + "_backup"
    if not os.path.exists(test_backup):
        shutil.copytree(test_dir, test_backup)
        print(f"Test backup created at: {test_backup}")
    
    # Move files
    moved = 0
    for f in to_move:
        src = os.path.join(train_dir, f)
        dst = os.path.join(test_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            moved += 1
    
    print(f"Moved {moved} winter samples from Train to Test")
    
    # New seasonal distribution
    for split, dir_path in [("Train", train_dir), ("Test", test_dir)]:
        seasons = {"winter": 0, "spring": 0, "summer": 0, "autumn": 0}
        radar_count = 0
        for f in os.listdir(dir_path):
            if f.endswith('.npz'):
                jd = extract_julian_day(f)
                seasons[get_season(jd)] += 1
                if has_radar_label(os.path.join(dir_path, f)):
                    radar_count += 1
        print(f"\nNew {split} distribution:")
        for s, count in seasons.items():
            print(f"  {s}: {count}")
        print(f"  With radar: {radar_count}")


if __name__ == "__main__":
    main()
