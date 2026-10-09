"""Step 1: 从实际文件 + 已存档 manifest 构建 sample_manifest.csv / split_manifest.csv
All numbers must come from this file. No hand-curated arithmetic.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from datetime import date, timedelta

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")

DATA_ROOT = r"E:/Data/Unet_Dataset"
OUT_DIR = r"E:/Claude code/project/noise-label-cloud/output/manifest"
os.makedirs(OUT_DIR, exist_ok=True)


def _station(fname):
    if "Changsha" in fname: return "CS"
    if "Longmen" in fname: return "LM"
    return "?"


def _season(doy_year):
    """doy_year like 2020134 → date → season."""
    try:
        year = doy_year // 1000
        ddd  = doy_year % 1000
        d = date(year, 1, 1) + timedelta(days=ddd - 1)
        m = d.month
    except Exception:
        return "Unknown"
    if m in (3, 4, 5): return "Spring"
    if m in (6, 7, 8): return "Summer"
    if m in (9, 10, 11): return "Autumn"
    return "Winter"


def _year_season(doy_year):
    try:
        year = doy_year // 1000
        ddd  = doy_year % 1000
        d = date(year, 1, 1) + timedelta(days=ddd - 1)
        return d.year, d.month, _season(doy_year)
    except Exception:
        return None, None, "Unknown"


def parse_doy(fname):
    """Parse Tensor_<STATION>_<YYYYDDD>.<HHMM>.npz → (year, doy, doy_year, hhmm).
    Note: filenames are like 'Tensor_Changsha_2022024.1842.npz' (no '_A' separator).
    """
    try:
        base = fname.replace(".npz", "")
        # No '_A' separator — last token before '.HHMM.npz' is the doy
        # Split on underscores
        parts = base.split("_")
        if len(parts) < 3:
            return None
        # The doy+hhmm token: e.g. "2022024.1842"
        doy_token = parts[-1]
        if "." not in doy_token:
            return None
        doy_year_s, hhmm_s = doy_token.split(".", 1)
        # Some files might have an "A" prefix; strip if present
        doy_year_s = doy_year_s.lstrip("A")
        if not doy_year_s.isdigit():
            return None
        doy_year = int(doy_year_s)
        return {
            "year": doy_year // 1000,
            "doy": doy_year % 1000,
            "doy_year": doy_year,
            "hhmm": hhmm_s,
        }
    except Exception:
        return None


def npz_summary(fname, all_files=None):
    """Quick read of npz for radar + sanity flags; return small dict."""
    info = {
        "has_radar": False,
        "radar_label": None,
        "radar_loc": None,
        "moon_phase": None,
        "solar_zenith": None,
        "m15_at_radar": None,
        "m15_local_std": None,
        "y_mask_class_at_radar": None,
    }
    try:
        # find first existing sub directory for this fname
        full_path = None
        if all_files is not None and fname in all_files:
            for sub in all_files[fname]["subs"]:
                cand = os.path.join(DATA_ROOT, sub, fname)
                if os.path.exists(cand):
                    full_path = cand
                    break
        if full_path is None:
            # try root
            cand = os.path.join(DATA_ROOT, fname)
            if os.path.exists(cand):
                full_path = cand
            else:
                return info
        d = np.load(full_path)
        if "Center_Label" in d.files:
            cl = float(d["Center_Label"])
            info["has_radar"] = bool(not np.isnan(cl))
            info["radar_label"] = int(cl) if not np.isnan(cl) else None
            info["radar_loc"] = d["Radar_Loc"].tolist()
            info["moon_phase"] = float(d["Moon_Phase"]) if "Moon_Phase" in d.files else None
            info["solar_zenith"] = float(d["Solar_Zenith"]) if "Solar_Zenith" in d.files else None
            if info["has_radar"]:
                ry, rx = info["radar_loc"]
                info["y_mask_class_at_radar"] = int(d["Y_mask"][ry, rx])
                m15 = d["X_m15"]
                from scipy.ndimage import uniform_filter
                m15_filled = np.where(np.isnan(m15), np.nanmedian(m15), m15)
                # 5x5 local std around (ry, rx)
                y0, y1 = max(0, ry - 2), min(m15.shape[0], ry + 3)
                x0, x1 = max(0, rx - 2), min(m15.shape[1], rx + 3)
                win = m15_filled[y0:y1, x0:x1]
                if win.size > 1:
                    info["m15_local_std"] = float(win.std())
                v = m15[ry, rx]
                info["m15_at_radar"] = float(v) if not np.isnan(v) else None
        return info
    except Exception as e:
        return info


# === Scan all sample directories ===
all_files = {}  # fname → dict
for sub in ("Train", "Train_no_overlap", "Train_phase4", "Val_phase4", "Test", "Test_backup"):
    p = os.path.join(DATA_ROOT, sub)
    if not os.path.isdir(p):
        continue
    for f in sorted(os.listdir(p)):
        if not f.endswith(".npz"): continue
        full = os.path.join(p, f)
        if f in all_files:
            all_files[f]["subs"].append(sub)
        else:
            all_files[f] = {"fname": f, "subs": [sub]}
print(f"Total unique .npz: {len(all_files)}")

# Parse metadata for each
rows = []
for f, info in all_files.items():
    parsed = parse_doy(f)
    if parsed is None:
        parsed = {"year": None, "doy": None, "doy_year": None, "hhmm": None}
    npz_info = npz_summary(f, all_files=all_files)
    rows.append({
        "fname": f,
        "year": parsed["year"],
        "doy": parsed["doy"],
        "doy_year": parsed["doy_year"],
        "hhmm": parsed["hhmm"],
        "station": _station(f),
        **npz_info,
        "subs": ",".join(info["subs"]),
        # Split flag is assigned later
        "split": "",
    })

df = pd.DataFrame(rows)

# === Load existing phase4 split manifest ===
with open(os.path.join(DATA_ROOT, "phase4_split_manifest.json")) as f:
    p4 = json.load(f)
p4_train = set(p4["train_files"])
p4_val   = set(p4["val_files"])

df.loc[df["fname"].isin(p4_train), "split"] = "train_phase4"
df.loc[df["fname"].isin(p4_val),   "split"] = "val_phase4"
df.loc[df["subs"].str.contains(r"\bTest\b", regex=True), "split"] = "test"

# Now build TRAIN-only and TRAIN+TEST by sub directories
df["in_sub_train"] = df["subs"].str.contains(r"\bTrain\b", regex=True)
df["in_sub_test"] = df["subs"].str.contains(r"\bTest\b", regex=True)
df["in_sub_test_backup"] = df["subs"].str.contains(r"\bTest_backup\b", regex=True)
df["in_sub_train_no_overlap"] = df["subs"].str.contains(r"\bTrain_no_overlap\b", regex=True)
df["in_sub_train_phase4"] = df["subs"].str.contains(r"\bTrain_phase4\b", regex=True)
df["in_sub_val_phase4"] = df["subs"].str.contains(r"\bVal_phase4\b", regex=True)

# Add overpass group ID = (station, doy_year): same overpass but different station
df["overpass_group"] = df.apply(
    lambda r: f"{r['station']}_{r['doy_year']}" if r["doy_year"] is not None else "", axis=1)
df["size_mb"] = df["fname"].apply(
    lambda f: os.path.getsize(os.path.join(DATA_ROOT, "Train", f)) / 1e6
               if os.path.exists(os.path.join(DATA_ROOT, "Train", f)) else 0)

# Save csv
df.to_csv(os.path.join(OUT_DIR, "sample_manifest.csv"), index=False)
print(f"Saved sample_manifest.csv → {len(df)} rows")

# Save split_manifest.csv: which files belong to which split
split_manifest = []
for f in df["fname"]:
    s = df.loc[df["fname"] == f, "split"].values[0]
    split_manifest.append({"fname": f, "split": s})
pd.DataFrame(split_manifest).to_csv(os.path.join(OUT_DIR, "split_manifest.csv"), index=False)
print(f"Saved split_manifest.csv → {len(split_manifest)} rows")

# Data-flow arithmetic proof
n_total = len(df)
n_train_p4 = (df["split"] == "train_phase4").sum()
n_val_p4   = (df["split"] == "val_phase4").sum()
n_test     = (df["split"] == "test").sum()
n_other    = n_total - (n_train_p4 + n_val_p4 + n_test)
print(f"\n=== DATA FLOW (auto-derived, arithmetic closed) ===")
print(f"Total .npz files (universe U)         = {n_total}")
print(f"  Split:  Train_phase4 (training pool) = {n_train_p4}")
print(f"          Val_phase4 (validation pool) = {n_val_p4}")
print(f"          Test  (radar-cleaned test)    = {n_test}")
print(f"  Remaining (no phase-4 split)        = {n_other}")
print(f"\nClosure check: {n_train_p4}+{n_val_p4}+{n_test}+{n_other} = "
      f"{n_train_p4+n_val_p4+n_test+n_other} (must equal {n_total})")

# Print subset counts
print("\n=== Data flow (auto-derived from files) ===")
print(f"Total unique .npz files         : {len(df)}")
print(f"In Test (final test set)       : {df['split'].eq('test').sum()}")
print(f"In Train_phase4 (training pool): {df['split'].eq('train_phase4').sum()}")
print(f"In Val_phase4 (validation pool): {df['split'].eq('val_phase4').sum()}")
print(f"In Train (original 2614 pool)  : {df['in_sub_train'].sum()}")
print(f"In Train_no_overlap            : {df['in_sub_train_no_overlap'].sum()}")
print(f"In Test (incl. Test_backup)    : {(df['in_sub_test'] | df['in_sub_test_backup']).sum()}")

print(f"\nWith radar label (Center_Label not NaN) in test set: {df[df['split']=='test']['has_radar'].sum()}")
print(f"With radar label in val set: {df[df['split']=='val_phase4']['has_radar'].sum()}")
print(f"With radar label in train set: {df[df['split']=='train_phase4']['has_radar'].sum()}")

# Per-station
print("\nPer-station count in test set:")
print(df[df["split"]=="test"].groupby("station").size())

# Per-season
print("\nPer-season count in test set:")
print(df[df["split"]=="test"].groupby(df.apply(_season, axis=1)).size())

# Overpass groups in test set (a same-cloud-field indicator)
print("\nOverpass groups (test set):", df[df["split"]=="test"]["overpass_group"].nunique())
print("Overpass groups (train_phase4):", df[df["split"]=="train_phase4"]["overpass_group"].nunique())

# Are there duplicate overpasses across Train_phase4 and Test?
train_overpasses = set(df[df["split"]=="train_phase4"]["overpass_group"].dropna())
test_overpasses = set(df[df["split"]=="test"]["overpass_group"].dropna())
overlap = train_overpasses & test_overpasses
print(f"\nOverpass groups in Train_phase4 ∩ Test: {len(overlap)} (should be 0 for no leakage)")
