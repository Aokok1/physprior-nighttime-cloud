import numpy as np
import os

DATA_DIR = r"E:\Data\Unet_Dataset"
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npz")])
print(f"Total files: {len(files)}")

# Check first file
fpath = os.path.join(DATA_DIR, files[0])
data = np.load(fpath, allow_pickle=True)
print(f"\nFile: {files[0]}")
print(f"Keys: {list(data.keys())}")

for k in data.keys():
    v = data[k]
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, ndim={v.ndim}")
    if v.ndim == 0:
        print(f"    value={v.item()}")
    elif v.size < 10:
        print(f"    values={v}")

# Check center pixel
y_mask = data["Y_mask"]
cy, cx = y_mask.shape[0] // 2, y_mask.shape[1] // 2
print(f"\nCenter pixel ({cy},{cx}):")
print(f"  Y_mask = {y_mask[cy, cx]}")

# Check Center_Label
cl = data["Center_Label"]
print(f"  Center_Label: shape={cl.shape}, dtype={cl.dtype}")
if cl.ndim == 0:
    print(f"  value={cl.item()}")
else:
    print(f"  center value={cl[cy, cx] if cl.shape == y_mask.shape else cl}")

# Check a few files for radar coverage
print(f"\nChecking first 10 files for radar coverage:")
for f in files[:10]:
    d = np.load(os.path.join(DATA_DIR, f), allow_pickle=True)
    cl = d["Center_Label"]
    if cl.ndim == 0:
        val = cl.item()
    else:
        cy2, cx2 = cl.shape[0]//2, cl.shape[1]//2
        val = cl[cy2, cx2]
    has_radar = not np.isnan(float(val))
    print(f"  {f}: Center_Label={val}, has_radar={has_radar}")
