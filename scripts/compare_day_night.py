"""
白天 vs 夜间 CLDMSK 准确率对比 —— 同一雷达像素，同一日期，不同 VIIRS 过境。

Nighttime CLDMSK: 来自 E:/Data/Unet_Dataset/ 的 .npz 文件 (已处理)
Daytime CLDMSK: 来自 E:/Data/daytime_cldmsk_test/ 的 .nc 文件 (新下载)

输出：白天 CLDMSK acc vs 夜间 CLDMSK acc，按站点、月相分段。
"""
import numpy as np
import h5py
import glob
import os
from datetime import datetime, timedelta
from collections import defaultdict


STATIONS = {
    "Changsha": (28.266, 113.036),
    "Longmen": (23.795, 114.275),
}
CLASS_NAMES = ["Clear", "ProbClear", "ProbCloud", "Cloud"]


def doy_to_date(yr, doy):
    return (datetime(yr, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")


def parse_filename_date(fname):
    """Extract year + day-of-year from CLDMSK filename.
    e.g. CLDMSK_L2_VIIRS_SNPP.A2020087.0500.001.X.nc → (2020, 87)
    """
    for part in fname.split("."):
        if part.startswith("A") and len(part) == 8:
            return int(part[1:5]), int(part[5:8])
    return None, None


def get_daytime_cldmsk(nc_path, station_lat, station_lon):
    """Extract CLDMSK class at nearest pixel to station."""
    with h5py.File(nc_path, "r") as f:
        lat = f["geolocation_data/latitude"][:]
        lon = f["geolocation_data/longitude"][:]
        icm = f["geophysical_data/Integer_Cloud_Mask"][:]

    dist = np.sqrt((lat - station_lat) ** 2 + (lon - station_lon) ** 2)
    min_idx = np.unravel_index(np.nanargmin(dist), dist.shape)
    cm_val = icm[min_idx]
    dist_deg = float(dist[min_idx])

    # If pixel is too far from station (>0.5 deg), granule doesn't cover it
    if dist_deg > 0.5:
        return None

    return int(cm_val)


def collect_nighttime_cldmsk():
    """Collect nighttime CLDMSK + radar labels from existing .npz files."""
    records = defaultdict(list)  # (date_key, station) → list of records

    for mode in ["Train", "Test"]:
        for fp in sorted(glob.glob(f"E:/Data/Unet_Dataset/{mode}/*.npz")):
            d = np.load(fp)
            cl = float(d["Center_Label"])
            if np.isnan(cl):
                continue

            fname = os.path.basename(fp)
            yr, doy = None, None
            for part in fname.split("_"):
                if part.startswith("A"):
                    yr = int(part[1:5])
                    doy = int(part[5:8])
                    break
            if yr is None:
                continue

            site = "Longmen" if "Longmen" in fname else "Changsha"
            ry, rx = int(d["Radar_Loc"][0]), int(d["Radar_Loc"][1])
            night_cm = int(d["Y_mask"][ry, rx])
            moon = float(d["Moon_Phase"])

            records[(yr, doy, site)].append({
                "radar": int(cl),
                "night_cldmsk": night_cm,
                "moon": moon,
                "file": fname,
            })

    return records


def main():
    night_records = collect_nighttime_cldmsk()
    print(f"Nighttime records: {sum(len(v) for v in night_records.values())} "
          f"from {len(night_records)} unique (date, site) keys")

    # Find daytime matches
    nc_dir = "E:/Data/daytime_cldmsk_test"
    nc_files = sorted(glob.glob(f"{nc_dir}/*.nc"))

    if not nc_files:
        print("No daytime NC files found. Run download first.")
        return

    print(f"Daytime NC files: {len(nc_files)}")

    # Compare
    matched = 0
    day_correct = 0
    night_correct = 0
    total = 0

    # Detailed results
    comparisons = []

    for nc_f in nc_files:
        fname = os.path.basename(nc_f)
        yr, doy = parse_filename_date(fname)
        if yr is None:
            continue

        for site, (slat, slon) in STATIONS.items():
            key = (yr, doy, site)
            if key not in night_records:
                continue

            day_cm = get_daytime_cldmsk(nc_f, slat, slon)
            if day_cm is None:
                continue

            for night_rec in night_records[key]:
                radar = night_rec["radar"]
                night_cm = night_rec["night_cldmsk"]
                moon = night_rec["moon"]

                # Binary classification: 0,1=clear; 2,3=cloud
                day_clear = day_cm < 2
                night_clear = night_cm < 2
                radar_clear = radar == 0

                day_correct_bool = (day_clear == radar_clear)
                night_correct_bool = (night_clear == radar_clear)

                comparisons.append({
                    "date": doy_to_date(yr, doy),
                    "site": site,
                    "radar": radar,
                    "day_cldmsk": day_cm,
                    "night_cldmsk": night_cm,
                    "moon": moon,
                    "day_correct": day_correct_bool,
                    "night_correct": night_correct_bool,
                })

                if day_correct_bool:
                    day_correct += 1
                if night_correct_bool:
                    night_correct += 1
                total += 1
                matched += 1

    print(f"\n{'='*60}")
    print(f"  Day vs Night CLDMSK Comparison")
    print(f"{'='*60}")
    print(f"  Matched comparisons: {matched}")

    if total > 0:
        day_acc = day_correct / total
        night_acc = night_correct / total

        print(f"\n  Daytime   CLDMSK accuracy: {day_acc:.2%} ({day_correct}/{total})")
        print(f"  Nighttime CLDMSK accuracy: {night_acc:.2%} ({night_correct}/{total})")
        print(f"  Difference: {day_acc - night_acc:+.1%}")

        # Statistical test
        from scipy.stats import binomtest
        night_ci = binomtest(night_correct, total)
        day_ci = binomtest(day_correct, total)
        print(f"\n  Night 95% CI: [{night_ci.proportion_ci()[0]:.1%}, "
              f"{night_ci.proportion_ci()[1]:.1%}]")
        print(f"  Day   95% CI: [{day_ci.proportion_ci()[0]:.1%}, "
              f"{day_ci.proportion_ci()[1]:.1%}]")

        # Per-site breakdown
        for site in ["Changsha", "Longmen"]:
            site_total = sum(1 for c in comparisons if c["site"] == site)
            site_day = sum(1 for c in comparisons if c["site"] == site and c["day_correct"])
            site_night = sum(1 for c in comparisons if c["site"] == site and c["night_correct"])
            if site_total > 0:
                print(f"\n  {site} (n={site_total}):")
                print(f"    Day:   {site_day/site_total:.1%}")
                print(f"    Night: {site_night/site_total:.1%}")

        # Per-moon-phase breakdown
        dark = [c for c in comparisons if c["moon"] < 60 or c["moon"] > 300]
        bright = [c for c in comparisons if 60 <= c["moon"] <= 300]
        for label, subset in [("Dark (moon 0-60 or 300-360)", dark),
                               ("Bright (moon 60-300)", bright)]:
            if subset:
                d = sum(1 for c in subset if c["day_correct"])
                n = sum(1 for c in subset if c["night_correct"])
                print(f"\n  {label} (n={len(subset)}):")
                print(f"    Day:   {d/len(subset):.1%}")
                print(f"    Night: {n/len(subset):.1%}")

        # Show first few comparisons as examples
        print(f"\n  Example comparisons:")
        for c in comparisons[:5]:
            print(f"    {c['date']} {c['site']}: "
                  f"Day={CLASS_NAMES[c['day_cldmsk']]} "
                  f"Night={CLASS_NAMES[c['night_cldmsk']]} "
                  f"Radar={'Clear' if c['radar']==0 else 'Cloud'} "
                  f"Moon={c['moon']:.0f}° "
                  f"[Day:{'OK' if c['day_correct'] else 'X'} Night:{'OK' if c['night_correct'] else 'X'}]")

        # Night false positives that were corrected by day
        fp_fixed = sum(1 for c in comparisons
                       if not c["night_correct"] and c["day_correct"])
        fn_introduced = sum(1 for c in comparisons
                            if c["night_correct"] and not c["day_correct"])
        print(f"\n  Night FP fixed by daytime: {fp_fixed}")
        print(f"  Night correct → Day wrong: {fn_introduced}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
