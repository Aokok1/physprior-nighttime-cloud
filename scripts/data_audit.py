"""Comprehensive data quality audit for noise-label project.

Checks:
1. npz file integrity (all required fields present)
2. Label distribution (CLDMSK 4-class, radar binary)
3. NaN/Inf in features
4. Value ranges (DNB, M15, Basemap)
5. Radar pixel location validity
6. Train/Test split overlap (temporal leakage)
7. Season distribution
8. Station distribution (Changsha vs Longmen)
"""
import os
import sys
import glob
import json
import numpy as np
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET

REQUIRED_FIELDS = ["X_dnb", "X_basemap", "Y_mask", "Y_height", "Center_Label", "Radar_Loc"]
OPTIONAL_FIELDS = ["X_m15", "X_mod", "Moon_Phase", "Solar_Zenith"]


def audit_single_file(filepath):
    """Audit a single npz file. Returns dict of findings."""
    issues = []
    info = {}
    
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        return {"error": str(e)}, {}
    
    filename = os.path.basename(filepath)
    info["filename"] = filename
    
    # 1. Check required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            issues.append(f"MISSING required field: {field}")
    
    if issues:
        return {"issues": issues}, info
    
    # 2. Extract metadata from filename
    import re
    station_match = re.search(r'Tensor_(\w+)_A', filename)
    if station_match:
        info["station"] = station_match.group(1)
    
    julian_match = re.search(r'A\d{4}(\d{3})\.', filename)
    if julian_match:
        info["julian_day"] = int(julian_match.group(1))
    
    # 3. Check shapes
    x_dnb = data["X_dnb"]
    x_basemap = data["X_basemap"]
    y_mask = np.squeeze(data["Y_mask"])
    y_height = data["Y_height"]
    center_label = data["Center_Label"]
    radar_loc = data["Radar_Loc"]
    
    info["dnb_shape"] = x_dnb.shape
    info["basemap_shape"] = x_basemap.shape
    info["mask_shape"] = y_mask.shape
    info["height_shape"] = y_height.shape
    
    expected_size = 128
    if x_dnb.shape != (expected_size, expected_size):
        issues.append(f"DNB shape {x_dnb.shape} != ({expected_size}, {expected_size})")
    if y_mask.shape != (expected_size, expected_size):
        issues.append(f"Mask shape {y_mask.shape} != ({expected_size}, {expected_size})")
    
    # 4. Check NaN/Inf
    for name, arr in [("DNB", x_dnb), ("Basemap", x_basemap), ("Height", y_height)]:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0:
            info[f"{name}_nan"] = int(nan_count)
        if inf_count > 0:
            info[f"{name}_inf"] = int(inf_count)
            issues.append(f"{name} has {inf_count} Inf values")
    
    # 5. Check value ranges
    info["dnb_range"] = [float(np.nanmin(x_dnb)), float(np.nanmax(x_dnb))]
    info["basemap_range"] = [float(np.nanmin(x_basemap)), float(np.nanmax(x_basemap))]
    
    # DNB should be non-negative (radiance)
    if np.nanmin(x_dnb) < 0:
        issues.append(f"DNB has negative values: min={np.nanmin(x_dnb)}")
    
    # 6. Check label distributions
    y_mask_int = y_mask.astype(int)
    unique_labels, label_counts = np.unique(y_mask_int, return_counts=True)
    info["label_distribution"] = {int(k): int(v) for k, v in zip(unique_labels, label_counts)}
    
    # Labels should be 0-3
    invalid_labels = [l for l in unique_labels if l < 0 or l > 3]
    if invalid_labels:
        issues.append(f"Invalid labels: {invalid_labels}")
    
    # 7. Check radar center label
    cl = float(center_label) if np.isscalar(center_label) else float(center_label.flat[0])
    info["center_label"] = cl
    if not np.isnan(cl) and cl not in [0.0, 1.0]:
        issues.append(f"Center_Label={cl} not in {{0, 1, NaN}}")
    
    # 8. Check radar location
    ry, rx = int(radar_loc[0]), int(radar_loc[1])
    info["radar_loc"] = [ry, rx]
    if ry < 0 or ry >= 128 or rx < 0 or rx >= 128:
        issues.append(f"Radar_Loc out of bounds: ({ry}, {rx})")
    
    # 9. Check CLDMSK at radar pixel
    if 0 <= ry < 128 and 0 <= rx < 128:
        cldmsk_at_radar = int(y_mask_int[ry, rx])
        info["cldmsk_at_radar"] = cldmsk_at_radar
    
    # 10. Optional fields
    if "X_m15" in data:
        x_m15 = data["X_m15"]
        info["m15_shape"] = x_m15.shape
        info["m15_range"] = [float(np.nanmin(x_m15)), float(np.nanmax(x_m15))]
    
    if "Moon_Phase" in data:
        info["moon_phase"] = float(data["Moon_Phase"])
    
    if "Solar_Zenith" in data:
        info["solar_zenith"] = float(data["Solar_Zenith"])
    
    return issues, info


def audit_split(split_dir, split_name):
    """Audit all files in a split (Train or Test)."""
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    
    if not files:
        return {"error": f"No files in {split_dir}"}
    
    all_issues = []
    all_info = []
    label_counter = Counter()
    station_counter = Counter()
    radar_label_counter = Counter()
    julian_days = []
    sza_values = []
    
    for f in files:
        issues, info = audit_single_file(f)
        if issues:
            all_issues.append({"file": os.path.basename(f), "issues": issues})
        all_info.append(info)
        
        # Aggregate stats
        if "label_distribution" in info:
            for k, v in info["label_distribution"].items():
                label_counter[k] += v
        
        if "station" in info:
            station_counter[info["station"]] += 1
        
        cl = info.get("center_label", float("nan"))
        if not np.isnan(cl):
            radar_label_counter[int(cl)] += 1
        
        if "julian_day" in info:
            julian_days.append(info["julian_day"])
        
        if "solar_zenith" in info:
            sza_values.append(info["solar_zenith"])
    
    # Compute stats
    results = {
        "split": split_name,
        "total_files": len(files),
        "files_with_issues": len(all_issues),
        "label_distribution": dict(label_counter),
        "station_distribution": dict(station_counter),
        "radar_label_distribution": dict(radar_label_counter),
    }
    
    if julian_days:
        results["julian_day_range"] = [min(julian_days), max(julian_days)]
        
        # Season distribution
        season_counter = Counter()
        for jd in julian_days:
            if jd >= 335 or jd < 60:
                season_counter["winter"] += 1
            elif jd < 152:
                season_counter["spring"] += 1
            elif jd < 244:
                season_counter["summer"] += 1
            else:
                season_counter["autumn"] += 1
        results["season_distribution"] = dict(season_counter)
    
    if sza_values:
        results["sza_range"] = [min(sza_values), max(sza_values)]
        daytime_count = sum(1 for s in sza_values if s < 90)
        nighttime_count = sum(1 for s in sza_values if s >= 90)
        results["day_night_split"] = {"daytime": daytime_count, "nighttime": nighttime_count}
    
    # Radar label validity check
    total_radar = sum(radar_label_counter.values())
    total_nan = len(files) - total_radar
    results["radar_coverage"] = {
        "with_radar": total_radar,
        "nan_radar": total_nan,
        "coverage_rate": total_radar / len(files) if files else 0,
    }
    
    if all_issues:
        results["issues_sample"] = all_issues[:10]  # First 10 issues
    
    return results


def check_temporal_leakage(train_dir, test_dir):
    """Check if train and test files have temporal overlap."""
    import re
    
    def extract_timestamps(directory):
        timestamps = []
        for f in glob.glob(os.path.join(directory, "*.npz")):
            fname = os.path.basename(f)
            # Extract: station + year + julian_day + time
            match = re.search(r'A(\d{4})(\d{3})\.(\d{4})', fname)
            if match:
                year, jd, time = int(match.group(1)), int(match.group(2)), match.group(3)
                timestamps.append((year, jd, time, fname))
        return timestamps
    
    train_ts = extract_timestamps(train_dir)
    test_ts = extract_timestamps(test_dir)
    
    train_set = set((t[0], t[1], t[2]) for t in train_ts)
    test_set = set((t[0], t[1], t[2]) for t in test_ts)
    
    overlap = train_set & test_set
    
    # Also check same-day overlap (same julian day)
    train_days = set((t[0], t[1]) for t in train_ts)
    test_days = set((t[0], t[1]) for t in test_ts)
    day_overlap = train_days & test_days
    
    results = {
        "train_files": len(train_ts),
        "test_files": len(test_ts),
        "exact_overlap": len(overlap),
        "same_day_overlap": len(day_overlap),
    }
    
    if overlap:
        results["exact_overlap_files"] = list(overlap)[:10]
    
    if day_overlap:
        results["same_day_overlap_days"] = sorted(list(day_overlap))[:20]
        # Check if same overpass has different times in train/test
        for year, jd in sorted(day_overlap):
            train_times = sorted([t[2] for t in train_ts if t[0]==year and t[1]==jd])
            test_times = sorted([t[2] for t in test_ts if t[0]==year and t[1]==jd])
            if train_times and test_times:
                results.setdefault("overlap_details", []).append({
                    "year": year, "julian_day": jd,
                    "train_times": train_times, "test_times": test_times,
                })
    
    return results


def main():
    """Run full data audit."""
    print("="*70)
    print("DATA QUALITY AUDIT — Noise-Label Cloud Detection")
    print("="*70)
    
    train_dir = os.path.join(MTUNET_DATASET, "Train")
    test_dir = os.path.join(MTUNET_DATASET, "Test")
    
    # 1. Audit Train split
    print("\n[1/4] Auditing Train split...")
    train_results = audit_split(train_dir, "Train")
    print(f"  Files: {train_results['total_files']}")
    print(f"  Issues: {train_results['files_with_issues']}")
    print(f"  Labels: {train_results['label_distribution']}")
    print(f"  Stations: {train_results['station_distribution']}")
    print(f"  Radar: {train_results['radar_label_distribution']}")
    print(f"  Coverage: {train_results['radar_coverage']}")
    if "season_distribution" in train_results:
        print(f"  Seasons: {train_results['season_distribution']}")
    if "day_night_split" in train_results:
        print(f"  Day/Night: {train_results['day_night_split']}")
    
    # 2. Audit Test split
    print("\n[2/4] Auditing Test split...")
    test_results = audit_split(test_dir, "Test")
    print(f"  Files: {test_results['total_files']}")
    print(f"  Issues: {test_results['files_with_issues']}")
    print(f"  Labels: {test_results['label_distribution']}")
    print(f"  Stations: {test_results['station_distribution']}")
    print(f"  Radar: {test_results['radar_label_distribution']}")
    print(f"  Coverage: {test_results['radar_coverage']}")
    if "season_distribution" in test_results:
        print(f"  Seasons: {test_results['season_distribution']}")
    
    # 3. Check temporal leakage
    print("\n[3/4] Checking temporal leakage...")
    leakage_results = check_temporal_leakage(train_dir, test_dir)
    print(f"  Train files: {leakage_results['train_files']}")
    print(f"  Test files: {leakage_results['test_files']}")
    print(f"  Exact overlap: {leakage_results['exact_overlap']}")
    print(f"  Same-day overlap: {leakage_results['same_day_overlap']}")
    
    if leakage_results['same_day_overlap'] > 0:
        print(f"  WARNING: Same-day overlap detected!")
        if "overlap_details" in leakage_results:
            for detail in leakage_results["overlap_details"][:5]:
                print(f"    Year {detail['year']}, JD {detail['julian_day']}: "
                      f"Train={detail['train_times']}, Test={detail['test_times']}")
    
    # 4. Cross-split consistency
    print("\n[4/4] Cross-split consistency check...")
    train_labels = train_results['label_distribution']
    test_labels = test_results['label_distribution']
    
    train_total = sum(train_labels.values())
    test_total = sum(test_labels.values())
    
    print(f"  Train label ratios: ", end="")
    for k in sorted(train_labels.keys()):
        print(f"  class_{k}: {train_labels[k]/train_total:.1%}", end="")
    print()
    
    print(f"  Test label ratios:  ", end="")
    for k in sorted(test_labels.keys()):
        print(f"  class_{k}: {test_labels[k]/test_total:.1%}", end="")
    print()
    
    # Save full report
    output_path = os.path.join(PROJECT_ROOT, "output", "data_audit_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    full_report = {
        "train": train_results,
        "test": test_results,
        "temporal_leakage": leakage_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
