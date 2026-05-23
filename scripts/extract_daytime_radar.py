"""
提取白天雷达云/晴空标签——用与白天 CLDMSK 同时刻的雷达观测作金标准。

Ka-band 雷达 24/7 连续工作。对每个白天 CLDMSK granule (~05:00 UTC)，
找到同时刻雷达廓线，用 dBZ 阈值判定该时刻是否有云。
"""
import numpy as np
import xarray as xr
import glob
import os
from datetime import datetime, timedelta
from collections import defaultdict


STATIONS = {
    "Changsha": "H:/code/Data/radar_data/Changsha/data_nc",
    "Longmen": "H:/code/Data/radar_data/Longmen/data_nc",
}


def radar_has_cloud(ds, target_time_utc, cloud_dbz_thresh=-20,
                    min_cloud_thickness_m=100, min_cloud_base_m=50):
    """
    判断 target_time_utc 时刻雷达站点上空是否有云。

    方法：找到与目标时间最近的雷达廓线。
    云判定标准（避免地面杂波误判）：
      - dBZ > cloud_dbz_thresh (-20 dBZ，排除微弱杂波)
      - 连续有效 bin 数对应的物理厚度 > min_cloud_thickness_m (100m)
      - 最低有效 bin 高度 > min_cloud_base_m (50m，排除地面杂波)
    """
    t = ds["time"].values
    h = ds.get("height", ds.get("Height")).values  # 高度 (m)
    idx = np.abs(t - np.datetime64(target_time_utc)).argmin()
    time_diff_min = abs((t[idx] - np.datetime64(target_time_utc)) / np.timedelta64(1, 'm'))

    if time_diff_min > 30:
        return None, time_diff_min

    z_profile = ds["Z"].values[:, idx]  # (n_height,)

    # 有效数据且高度 > 地面杂波区
    valid = ~np.isnan(z_profile) & (h > min_cloud_base_m)
    if valid.sum() == 0:
        return 0, time_diff_min  # no valid data above clutter → assume clear

    # 连续云层检测
    bins_cloud = (z_profile > cloud_dbz_thresh) & valid
    if not bins_cloud.any():
        return 0, time_diff_min

    # 找最长的连续云段
    cloud_top = h[bins_cloud].max()
    cloud_bot = h[bins_cloud].min()
    thickness = cloud_top - cloud_bot

    if thickness < min_cloud_thickness_m:
        return 0, time_diff_min

    return 1, time_diff_min


def build_radar_index():
    """
    构建雷达文件索引：{site: {date_key: filepath}}
    雷达文件名格式：Changsha_KA_20220118.nc / Longmen_KA_20200327.nc
    """
    index = {}
    for site, dirpath in STATIONS.items():
        site_index = {}
        for fp in sorted(glob.glob(f"{dirpath}/*.nc")):
            fname = os.path.basename(fp)
            # Parse: {Site}_KA_{YYYYMMDD}.nc
            parts = fname.replace(".nc", "").split("_")
            date_str = parts[-1]  # YYYYMMDD
            if len(date_str) == 8:
                site_index[date_str] = fp
        index[site] = site_index
    return index


def get_daytime_radar_label(radar_index, site, yr, doy, daytime_utc_hour=5):
    """
    Get radar cloud label at the daytime VIIRS overpass time.

    daytime_utc_hour: expected VIIRS daytime overpass UTC hour (~5 for Changsha/Longmen)
    """
    date_str = f"{yr}{doy:03d}"
    # First try exact date match (YYYMMDD from radar filename)
    from datetime import datetime as dt
    date_ymd = (dt(yr, 1, 1) + timedelta(days=doy - 1)).strftime("%Y%m%d")

    site_index = radar_index.get(site, {})
    fp = site_index.get(date_ymd)
    if fp is None:
        return None, "no_radar_file"

    try:
        ds = xr.open_dataset(fp)
        target_time = f"{(dt(yr,1,1)+timedelta(days=doy-1)).strftime('%Y-%m-%d')}T{daytime_utc_hour:02d}:00:00"
        has_cloud, dt_min = radar_has_cloud(ds, target_time)
        ds.close()
        return has_cloud, dt_min
    except Exception as e:
        return None, str(e)[:50]


def main():
    import h5py
    from scipy.spatial import KDTree

    print("Building radar index...")
    radar_index = build_radar_index()
    for site, idx in radar_index.items():
        print(f"  {site}: {len(idx)} radar files")

    nc_dir = "E:/Data/daytime_cldmsk"
    if not os.path.exists(nc_dir) or len(os.listdir(nc_dir)) < 2:
        nc_dir = "E:/Data/daytime_cldmsk_test"
    nc_files = sorted(glob.glob(f"{nc_dir}/*.nc"))
    print(f"Daytime CLDMSK files: {len(nc_files)}")

    results = []
    matched = skipped_no_radar = skipped_no_match = 0

    for nc_f in nc_files:
        fname = os.path.basename(nc_f)
        yr = doy = None
        for part in fname.split("."):
            if part.startswith("A20") and len(part) == 8:
                yr = int(part[1:5])
                doy = int(part[5:8])
                break
        if yr is None:
            continue

        try:
            with h5py.File(nc_f, "r") as f:
                lat = f["geolocation_data/latitude"][:]
                lon = f["geolocation_data/longitude"][:]
                icm = f["geophysical_data/Integer_Cloud_Mask"][:]

            for site, (slat, slon) in [("Changsha", (28.266, 113.036)),
                                         ("Longmen", (23.795, 114.275))]:
                dist = np.sqrt((lat - slat) ** 2 + (lon - slon) ** 2)
                day_cm = int(icm[np.unravel_index(np.nanargmin(dist), dist.shape)])

                has_cloud, info = get_daytime_radar_label(radar_index, site, yr, doy)
                if has_cloud is None:
                    skipped_no_radar += 1
                    continue

                # Binary: day_cm 0,1=clear 2,3=cloud
                day_clr = day_cm < 2
                radar_clr = has_cloud == 0
                correct = day_clr == radar_clr

                results.append({
                    "date": f"{yr}-{doy:03d}",
                    "site": site,
                    "day_cldmsk": day_cm,
                    "radar_cloud": has_cloud,
                    "time_diff_min": info if isinstance(info, (int, float)) else 999,
                    "correct": correct,
                })
                matched += 1

        except Exception as e:
            skipped_no_match += 1

    print(f"\nMatched: {matched}, skipped (no radar): {skipped_no_radar}, "
          f"skipped (error): {skipped_no_match}")

    if results:
        correct = sum(r["correct"] for r in results)
        acc = correct / len(results)
        # CI
        from scipy.stats import binomtest
        ci = binomtest(correct, len(results)).proportion_ci()

        print(f"\n{'='*60}")
        print(f"  Daytime CLDMSK vs Daytime Radar")
        print(f"{'='*60}")
        print(f"  Accuracy: {acc:.1%} ({correct}/{len(results)})")
        print(f"  95% CI:   [{ci[0]:.1%}, {ci[1]:.1%}]")

        # Per-site
        for site in ["Changsha", "Longmen"]:
            site_r = [r for r in results if r["site"] == site]
            if site_r:
                sc = sum(r["correct"] for r in site_r)
                print(f"  {site}: {sc/len(site_r):.1%} ({sc}/{len(site_r)})")

        # Show examples
        print(f"\n  First 10 comparisons:")
        for r in results[:10]:
            cm_names = ["Clear", "ProbClear", "ProbCloud", "Cloud"]
            print(f"    {r['date']} {r['site']}: CLDMSK={cm_names[r['day_cldmsk']]} "
                  f"Radar={'Cloud' if r['radar_cloud'] else 'Clear'} "
                  f"dt={r['time_diff_min']:.0f}min "
                  f"[{'OK' if r['correct'] else 'X'}]")

    return results


if __name__ == "__main__":
    main()
