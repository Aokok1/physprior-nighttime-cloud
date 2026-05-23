"""Phase 3a: Identify and remove same-overpass samples from training set.

113 samples in Train (Changsha) share the same satellite overpass timestamp
with samples in Test (Longmen). This creates spatial correlation that may
inflate test accuracy.

This script:
1. Identifies the 113 overlapping timestamps
2. Creates a new Train split without these samples
3. Reports statistics
"""
import os
import sys
import re
import shutil
import json
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET


def extract_timestamp(filename):
    """Extract (year, julian_day, time) from filename."""
    match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)), match.group(3))
    return None


def extract_station(filename):
    """Extract station name from filename."""
    match = re.search(r'Tensor_(\w+)_A', filename)
    if match:
        return match.group(1)
    return None


def find_overlapping_samples():
    """Find Train samples that share overpass timestamp with Test samples."""
    train_dir = os.path.join(MTUNET_DATASET, "Train")
    test_dir = os.path.join(MTUNET_DATASET, "Test")
    
    # Build timestamp index for test
    test_timestamps = {}
    for f in os.listdir(test_dir):
        if f.endswith('.npz'):
            ts = extract_timestamp(f)
            if ts:
                test_timestamps[ts] = f
    
    # Find train samples with same timestamp
    overlapping = []
    for f in os.listdir(train_dir):
        if f.endswith('.npz'):
            ts = extract_timestamp(f)
            if ts and ts in test_timestamps:
                train_station = extract_station(f)
                test_station = extract_station(test_timestamps[ts])
                if train_station != test_station:
                    overlapping.append({
                        'filename': f,
                        'timestamp': ts,
                        'station': train_station,
                        'test_file': test_timestamps[ts],
                        'test_station': test_station,
                    })
    
    return overlapping


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be removed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for filtered train set")
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 3a: Identify Same-Overpass Samples")
    print("="*60)
    
    overlapping = find_overlapping_samples()
    
    print(f"\nFound {len(overlapping)} overlapping samples")
    print(f"\nSample overlaps:")
    for item in overlapping[:10]:
        print(f"  Train: {item['filename']} ({item['station']})")
        print(f"  Test:  {item['test_file']} ({item['test_station']})")
        print()
    
    # Group by station
    by_station = defaultdict(list)
    for item in overlapping:
        by_station[item['station']].append(item['filename'])
    
    print("By station:")
    for station, files in by_station.items():
        print(f"  {station}: {len(files)} files")
    
    if args.dry_run:
        print("\n[DRY RUN] No files modified")
        return
    
    # Create filtered train directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(MTUNET_DATASET, "Train_no_overlap")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_dir = os.path.join(MTUNET_DATASET, "Train")
    overlap_files = set(item['filename'] for item in overlapping)
    
    copied = 0
    skipped = 0
    for f in os.listdir(train_dir):
        if f.endswith('.npz'):
            if f not in overlap_files:
                src = os.path.join(train_dir, f)
                dst = os.path.join(output_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                copied += 1
            else:
                skipped += 1
    
    print(f"\nFiltered train set created at: {output_dir}")
    print(f"  Copied: {copied} files")
    print(f"  Removed: {skipped} files (same-overpass)")
    
    # Save overlap info
    info_path = os.path.join(output_dir, "overlap_info.json")
    with open(info_path, "w") as f:
        json.dump({
            "total_overlapping": len(overlapping),
            "by_station": {k: len(v) for k, v in by_station.items()},
            "overlapping_files": [item['filename'] for item in overlapping],
        }, f, indent=2)
    print(f"  Overlap info saved to: {info_path}")


if __name__ == "__main__":
    main()
