"""Trace data pipeline - part 2: source data locations"""
import os

print("=" * 60)
print(" Source Data Locations")
print("=" * 60)

for path in [
    "E:/Data/",
    "E:/Data/Unet_Dataset/",
    "E:/Data/NC_area/",
]:
    if os.path.exists(path):
        items = os.listdir(path)
        print(f"\n{path}: {len(items)} items")
        for item in sorted(items)[:15]:
            full = os.path.join(path, item)
            if os.path.isdir(full):
                try:
                    sub = os.listdir(full)
                    print(f"  [DIR] {item}/ ({len(sub)} files)")
                except:
                    print(f"  [DIR] {item}/ (access denied)")
            else:
                size = os.path.getsize(full)
                print(f"  [FILE] {item} ({size/1024/1024:.1f} MB)")
    else:
        print(f"\n{path}: NOT ACCESSIBLE")

# Check H drive
print("\n" + "=" * 60)
print(" H Drive (USB)")
print("=" * 60)
h_paths = [
    "H:/code/Data/",
    "H:/code/server_package/data/",
]
for path in h_paths:
    if os.path.exists(path):
        items = os.listdir(path)
        print(f"\n{path}: {len(items)} items")
        for item in sorted(items)[:10]:
            full = os.path.join(path, item)
            if os.path.isdir(full):
                try:
                    sub = os.listdir(full)
                    print(f"  [DIR] {item}/ ({len(sub)} files)")
                except:
                    print(f"  [DIR] {item}/")
            else:
                size = os.path.getsize(full)
                print(f"  [FILE] {item} ({size/1024/1024:.1f} MB)")
    else:
        print(f"\n{path}: NOT ACCESSIBLE (H: drive may not be mounted)")

# Check if there are radar files
print("\n" + "=" * 60)
print(" Radar Source Files")
print("=" * 60)
radar_paths = [
    "H:/code/Data/radar_data/",
    "H:/code/server_package/data/radar/",
]
for path in radar_paths:
    if os.path.exists(path):
        items = os.listdir(path)
        print(f"\n{path}: {len(items)} items")
        for item in sorted(items)[:10]:
            print(f"  {item}")
    else:
        print(f"\n{path}: NOT ACCESSIBLE")
