"""
精准查询白天 CLDMSK 数据——只下载与雷达日期匹配的 granules。
每个日期在 04:00-07:00 UTC（白天 VIIRS 过境窗口）内查询。

输出: nasa_daytime_cldmsk.txt（下载 URL 列表）
"""
import requests
import numpy as np
import glob
import os
from datetime import datetime, timedelta
from collections import defaultdict

# ── 1. 提取所有雷达匹配日期的列表 ──────────────────────────────
DATASET = r"E:/Data/Unet_Dataset"

def get_radar_dates():
    """从 .npz 文件中提取所有 Center_Label != NaN 的日期"""
    radar_dates = []
    for mode in ["Train", "Test"]:
        for fp in sorted(glob.glob(f"{DATASET}/{mode}/*.npz")):
            d = np.load(fp)
            cl = float(d["Center_Label"])
            if np.isnan(cl):
                continue
            fname = os.path.basename(fp)
            for part in fname.split("_"):
                if part.startswith("A"):
                    yr = int(part[1:5])
                    doy = int(part[5:8])
                    radar_dates.append((yr, doy))
                    break
    return sorted(set(radar_dates))


def doy_to_date(yr, doy):
    """Convert year + day-of-year to YYYY-MM-DD."""
    return (datetime(yr, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")


# ── 2. 按月份分组，逐月查询 ────────────────────────────────────
def query_cmr_for_month(yr, month, dates_in_month):
    """
    查询 CMR API 获取指定月份内所有雷达日期的白天 CLDMSK granules。

    CMR temporal 参数用月范围（避免逐日查询 344 次），
    然后按 bbox + day_night_flag 过滤。
    """
    # Month start/end
    start_dt = datetime(yr, month, 1)
    if month == 12:
        end_dt = datetime(yr + 1, 1, 1) - timedelta(seconds=1)
    else:
        end_dt = datetime(yr, month + 1, 1) - timedelta(seconds=1)

    # VIIRS daytime pass for this area: ~05:00-06:30 UTC
    # Query each day individually to be precise
    all_links = []

    for (yr_d, doy_d) in dates_in_month:
        date_str = doy_to_date(yr_d, doy_d)
        # Daytime VIIRS window: 04:00-07:00 UTC covers the ~5:30 UTC pass
        temporal = f"{date_str}T04:00:00Z,{date_str}T07:00:00Z"

        params = {
            "short_name": "CLDMSK_L2_VIIRS_SNPP",
            "bounding_box": "109.7,22.8,115.1,32.2",
            "temporal": temporal,
            "day_night_flag": "DAY",
            "page_size": 50,
            "page_num": 1,
        }

        url_api = "https://cmr.earthdata.nasa.gov/search/granules.json"

        try:
            resp = requests.get(url_api, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"    {date_str}: HTTP {resp.status_code}")
                continue

            data = resp.json()
            granules = data.get("feed", {}).get("entry", [])

            for g in granules:
                for link in g.get("links", []):
                    href = link.get("href", "")
                    rel = link.get("rel", "")
                    if "data#" in rel and ".nc" in href:
                        all_links.append(href)
                        break

            if granules:
                print(f"    {date_str}: {len(granules)} granules found")

        except requests.exceptions.Timeout:
            print(f"    {date_str}: TIMEOUT")
        except Exception as e:
            print(f"    {date_str}: ERROR {e}")

    return all_links


# ── 3. 主函数 ──────────────────────────────────────────────────
def main(max_dates=None):
    """
    max_dates: 限制查询日期数（None=全部 344 天，30=POC 测试）
    """
    radar_dates = get_radar_dates()
    print(f"雷达匹配日期: {len(radar_dates)} 天")

    if max_dates:
        radar_dates = radar_dates[:max_dates]
        print(f"限制查询前 {max_dates} 天（POC 测试）")

    # Group by (year, month)
    by_month = defaultdict(list)
    for yr, doy in radar_dates:
        month = (datetime(yr, 1, 1) + timedelta(days=doy - 1)).month
        by_month[(yr, month)].append((yr, doy))

    print(f"覆盖 {len(by_month)} 个月份\n")

    all_links = []
    for (yr, month), dates in sorted(by_month.items()):
        print(f"查询 {yr}-{month:02d} ({len(dates)} 天)...")
        links = query_cmr_for_month(yr, month, dates)
        all_links.extend(links)
        print(f"  本月获取 {len(links)} 个下载链接 (累计 {len(all_links)})")

    # 写入文件
    output_file = "nasa_daytime_cldmsk.txt"
    with open(output_file, "w") as f:
        for link in all_links:
            f.write(link + "\n")

    print(f"\n{'='*60}")
    print(f"完成！共 {len(all_links)} 个白天 CLDMSK 下载链接")
    print(f"已保存至: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc", type=int, default=30,
                        help="POC 测试：只查询前 N 个日期 (default: 30)")
    parser.add_argument("--all", action="store_true",
                        help="查询全部 344 个雷达日期")
    args = parser.parse_args()

    if args.all:
        main(max_dates=None)
    else:
        main(max_dates=args.poc)
