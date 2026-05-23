"""
并行下载白天 CLDMSK（参考 download2.py 的 ThreadPoolExecutor 模式）。
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

TOKEN = 'eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImNoYW5nd2FuZ2xvdTIwMSIsImV4cCI6MTc4MDQ4OTAwNiwiaWF0IjoxNzc1MzA1MDA2LCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.dtBxCT8vu3JpvTEhHpX8E18pdpXXEGl20nj7IpdjtmCOfVlyXFfKqr8-n_zg6fHny9nAoVrG7K7_qRTIVmEtrkgKF5gvp71niWbdMFAnBFvzVlyZ5580BgAsGQFycmAKprLLbDjOvoJMPsZtXBouDhxOX0J6b0vuWkyxAcA9NiK-S7WTA3X6n1sGqYw5qJTLULBvha91KeXH3ekLCmVPrEuwoaxvbTQIiHitTuRrK3xHBC072m2Xt2xxJJfmmPNvYPipG6m8ddTTtqvO4gZmksgfoUnAp108EvHOr1vT_7EY965d2Q0uiq4sQZ96n8q8JjsCzS_mztPivhIB-MDYeA'
DOWNLOAD_DIR = r"E:\Data\daytime_cldmsk"
LOG_FILE = r"E:\Data\daytime_cldmsk\download_log.txt"


def download_file(url, max_retries=3):
    filename = url.split('/')[-1]
    dest = os.path.join(DOWNLOAD_DIR, filename)

    if os.path.exists(dest) and os.path.getsize(dest) > 100000:
        return f"SKIP: {filename}", filename, True

    command = [
        "wget", "-c", "-q",
        "-T", "30", "-t", "3",
        url,
        "--header", f"Authorization: Bearer {TOKEN}",
        "-P", DOWNLOAD_DIR
    ]

    for attempt in range(max_retries):
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
            return f"OK: {filename}", filename, True
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT ({attempt+1}/{max_retries}): {filename}")
        except subprocess.CalledProcessError as e:
            return f"FAIL: {filename} (rc={e.returncode})", filename, False
        except Exception as e:
            return f"ERROR: {filename} ({e})", filename, False

    return f"GAVE UP: {filename}", filename, False


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Read URL list (from CMR query)
    url_file = r"E:\Claude code\project\noise-label-cloud\scripts\nasa_daytime_cldmsk.txt"
    if not os.path.exists(url_file):
        print(f"URL list not found: {url_file}")
        return

    with open(url_file) as f:
        all_urls = [l.strip() for l in f if l.strip()]

    # Filter: prefer .001 collection (standard processing)
    urls = [u for u in all_urls if '.001.' in u]
    print(f"Filtered: {len(urls)} .001 granules from {len(all_urls)} total")

    # Read completed log
    completed = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            completed = set(l.strip() for l in f if l.strip())
    print(f"Already completed: {len(completed)}")

    pending = [u for u in urls if u.split('/')[-1] not in completed]
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All done!")
        return

    max_workers = 4
    print(f"Starting parallel download with {max_workers} workers...\n")

    with open(LOG_FILE, "a") as log_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_file, url): url for url in pending}
            for future in as_completed(futures):
                msg, filename, is_success = future.result()
                print(f"  {msg}")
                if is_success:
                    log_f.write(filename + "\n")
                    log_f.flush()

    n_done = len(os.listdir(DOWNLOAD_DIR)) - 1  # minus LOG_FILE
    print(f"\nDone. {n_done} files in {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
