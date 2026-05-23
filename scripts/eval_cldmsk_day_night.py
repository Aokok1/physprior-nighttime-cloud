"""
Evaluate daytime and nighttime CLDMSK against Ka-band radar.
Outputs per-station and combined accuracy with 95% CI.
"""
import numpy as np
import os, glob, sys
from datetime import datetime, timedelta
from scipy.stats import binomtest

# ============== Part 1: Daytime CLDMSK vs Radar ==============
print("=" * 60)
print("PART 1: Daytime CLDMSK vs Radar")
print("=" * 60)

DAYTIME_DIR = r"E:\Data\daytime_cldmsk"
RADAR_DIRS = {
    "Changsha": r"H:\code\Data\radar_data\Changsha\data_nc",
    "Longmen": r"H:\code\Data\radar_data\Longmen\data_nc",
}
STATIONS = {
    "Changsha": (28.266, 113.036),
    "Longmen": (23.795, 114.275),
}

def find_radar_file(radar_dir, date_ymd):
    """Find radar nc file for a given date (YYYYMMDD)."""
    for fname in os.listdir(radar_dir):
        if date_ymd in fname and fname.endswith('.nc'):
            return os.path.join(radar_dir, fname)
    return None

def radar_has_cloud(fp, target_dt, cloud_dbz=-20, min_thickness=100, min_base=50):
    """Check if radar shows cloud at target datetime."""
    import xarray as xr
    try:
        ds = xr.open_dataset(fp)
        t = ds["time"].values
        h = ds.get("height", ds.get("Height")).values
        idx = np.abs(t - np.datetime64(target_dt)).argmin()
        time_diff = abs((t[idx] - np.datetime64(target_dt)) / np.timedelta64(1, 'm'))
        if time_diff > 30:
            ds.close()
            return None, time_diff
        
        z = ds["Z"].values[:, idx]
        valid = ~np.isnan(z) & (h > min_base)
        if valid.sum() == 0:
            ds.close()
            return 0, time_diff
        
        cloud_bins = (z > cloud_dbz) & valid
        if not cloud_bins.any():
            ds.close()
            return 0, time_diff
        
        cloud_top = h[cloud_bins].max()
        cloud_bot = h[cloud_bins].min()
        thickness = cloud_top - cloud_bot
        ds.close()
        return 1 if thickness >= min_thickness else 0, time_diff
    except:
        return None, 999

# Build radar file index
radar_index = {}
for site, rdir in RADAR_DIRS.items():
    try:
        files = [f for f in os.listdir(rdir) if f.endswith('.nc')]
        radar_index[site] = {f.split('_')[-1].replace('.nc', ''): os.path.join(rdir, f) for f in files}
        print(f"  {site}: {len(radar_index[site])} radar files")
    except Exception as e:
        print(f"  {site}: SKIP - {e}")
        radar_index[site] = {}

daytime_results = []
skipped_no_radar = 0
skipped_error = 0
matched = 0

nc_files = sorted(glob.glob(os.path.join(DAYTIME_DIR, "*.nc")))
print(f"  Daytime CLDMSK granules: {len(nc_files)}")

for nc_f in nc_files:
    fname = os.path.basename(nc_f)
    # Parse AYYYYDDD
    parts = fname.split('.')
    yr = doy = None
    for p in parts:
        if p.startswith('A20') and len(p) == 8:
            yr = int(p[1:5])
            doy = int(p[5:8])
            break
    if yr is None:
        continue
    
    date_dt = datetime(yr, 1, 1) + timedelta(days=doy - 1)
    date_ymd = date_dt.strftime('%Y%m%d')
    
    # Read CLDMSK L2
    import h5py
    try:
        with h5py.File(nc_f, 'r') as f:
            lat = f["geolocation_data/latitude"][:]
            lon = f["geolocation_data/longitude"][:]
            icm = f["geophysical_data/Integer_Cloud_Mask"][:]
    except:
        skipped_error += 1
        continue
    
    # Determine UTC hour from filename (e.g. .0500. = 05:00 UTC)
    utc_hour = 5  # default daytime overpass
    for p in parts:
        if len(p) == 4 and p.isdigit():
            utc_hour = int(p[:2])
            break
    
    target_dt = datetime(yr, date_dt.month, date_dt.day, utc_hour, 0, 0)
    
    for site, (slat, slon) in STATIONS.items():
        if site not in radar_index:
            continue
        # Find nearest pixel
        dist = np.sqrt((lat - slat)**2 + (lon - slon)**2)
        min_idx = np.unravel_index(np.nanargmin(dist), dist.shape)
        day_cm = int(icm[min_idx])
        
        # Find radar file
        fp = radar_index[site].get(date_ymd)
        if fp is None:
            skipped_no_radar += 1
            continue
        
        has_cloud, info = radar_has_cloud(fp, target_dt)
        if has_cloud is None:
            skipped_no_radar += 1
            continue
        
        day_clear = day_cm < 2  # 0,1 = clear; 2,3 = cloud
        radar_clear = has_cloud == 0
        correct = day_clear == radar_clear
        matched += 1
        
        daytime_results.append({
            "date": f"{yr}-{doy:03d}",
            "site": site,
            "day_cldmsk": day_cm,
            "radar_cloud": has_cloud,
            "correct": correct,
        })

print(f"  Matched: {matched}, skipped (no radar): {skipped_no_radar}, skipped (error): {skipped_error}")

if daytime_results:
    correct = sum(r["correct"] for r in daytime_results)
    acc = correct / len(daytime_results)
    ci = binomtest(correct, len(daytime_results)).proportion_ci()
    print(f"\n  >>> Daytime CLDMSK vs Radar <<<")
    print(f"  Combined: {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]  (n={len(daytime_results)})")
    for site in ["Changsha", "Longmen"]:
        sr = [r for r in daytime_results if r["site"] == site]
        if sr:
            sc = sum(r["correct"] for r in sr)
            sci = binomtest(sc, len(sr)).proportion_ci()
            print(f"  {site}: {sc/len(sr):.1%} [{sci[0]:.1%}, {sci[1]:.1%}]  (n={len(sr)})")
else:
    print("  WARNING: No daytime matches found!")

# ============== Part 2: Nighttime CLDMSK vs Radar ==============
print("\n" + "=" * 60)
print("PART 2: Nighttime CLDMSK vs Radar")
print("=" * 60)

# Use Val + Test .npz files from MT-UNet dataset
NPZ_DIRS = [
    r"H:\code\server_package\data\Unet_Dataset\Val",
    r"H:\code\server_package\data\Unet_Dataset\Test",
]

night_results = []
for ddir in NPZ_DIRS:
    if not os.path.exists(ddir):
        print(f"  SKIP: {ddir} not found")
        continue
    files = sorted(glob.glob(os.path.join(ddir, "*.npz")))
    print(f"  {os.path.basename(ddir)}: {len(files)} files")
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        mask = d["Y_mask"]  # 128x128, 0-3
        radar_label = float(d["Center_Label"])
        if np.isnan(radar_label):
            continue
        
        # CLDMSK center pixel
        cldmsk_val = mask[64, 64]
        cldmsk_cloud = cldmsk_val >= 2  # Probably Cloudy or True Cloud
        radar_cloud = radar_label == 1
        
        # Determine station
        fname = os.path.basename(fp)
        site = "Changsha" if "Changsha" in fname else "Longmen"
        
        correct = cldmsk_cloud == radar_cloud
        night_results.append({
            "file": fname,
            "site": site,
            "cldmsk_val": int(cldmsk_val),
            "radar_label": int(radar_label),
            "correct": correct,
        })

if night_results:
    correct = sum(r["correct"] for r in night_results)
    acc = correct / len(night_results)
    ci = binomtest(correct, len(night_results)).proportion_ci()
    print(f"\n  >>> Nighttime CLDMSK vs Radar <<<")
    print(f"  Combined: {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]  (n={len(night_results)})")
    for site in ["Changsha", "Longmen"]:
        sr = [r for r in night_results if r["site"] == site]
        if sr:
            sc = sum(r["correct"] for r in sr)
            sci = binomtest(sc, len(sr)).proportion_ci()
            print(f"  {site}: {sc/len(sr):.1%} [{sci[0]:.1%}, {sci[1]:.1%}]  (n={len(sr)})")
    
    # Also compute True Clear recall (how many actually-clear pixels are called clear)
    true_clear = [r for r in night_results if r["radar_label"] == 0]
    if true_clear:
        tc_correct = sum(r["correct"] for r in true_clear)
        print(f"  True Clear recall: {tc_correct/len(true_clear):.1%} ({tc_correct}/{len(true_clear)})")
    
    # Cloud recall
    true_cloud = [r for r in night_results if r["radar_label"] == 1]
    if true_cloud:
        tcl_correct = sum(r["correct"] for r in true_cloud)
        print(f"  True Cloud recall: {tcl_correct/len(true_cloud):.1%} ({tcl_correct}/{len(true_cloud)})")
else:
    print("  WARNING: No nighttime matches found!")

# ============== Summary ==============
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if daytime_results:
    d_correct = sum(r["correct"] for r in daytime_results)
    print(f"  Daytime:   {d_correct/len(daytime_results):.1%} (n={len(daytime_results)})")
if night_results:
    n_correct = sum(r["correct"] for r in night_results)
    print(f"  Nighttime: {n_correct/len(night_results):.1%} (n={len(night_results)})")
if daytime_results and night_results:
    print(f"  Day-Night degradation: {(d_correct/len(daytime_results) - n_correct/len(night_results))*100:.1f} pp")
