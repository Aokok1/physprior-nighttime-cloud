"""Cross-validate: read .nc file's Integer_Cloud_Mask values, compare to npz Y_mask.
This is the stop-the-line check for CLDMSK category mapping.
"""
import netCDF4 as nc
import numpy as np
import os, glob

# Pick the same data date as the npz test set
nc_files = sorted(glob.glob('I:/global_dnb_cloud/cldmsk_nc_v3/CLDMSK_L2_VIIRS_SNPP.A2022024.18*.nc'))
print(f'Found {len(nc_files)} nc files for 2022024 ~ 18:')
for f in nc_files[:3]:
    print(f'  {os.path.basename(f)}')

if not nc_files:
    print('No matching nc files; check date format')
    raise SystemExit

# Find a corresponding npz test file
npz_dir = 'E:/Data/Unet_Dataset/Test'
npz_files = sorted(glob.glob(f'{npz_dir}/Tensor_Changsha_A2022024*.npz'))
print(f'\nFound {len(npz_files)} npz test files for Changsha 2022024')
for f in npz_files[:3]:
    print(f'  {os.path.basename(f)}')

if not npz_files:
    print('No npz files; cannot cross-validate')
    raise SystemExit

# Open the npz
npz_path = npz_files[0]
npz = np.load(npz_path)
y_mask = npz['Y_mask']
print(f'\nNPY file: {os.path.basename(npz_path)}')
print(f'  Y_mask shape: {y_mask.shape}, dtype: {y_mask.dtype}')
print(f'  Y_mask unique values: {np.unique(y_mask)}')
print(f'  Y_mask counts: {dict(zip(*np.unique(y_mask, return_counts=True)))}')
print(f'  radar_loc: {npz.get("radar_loc", "none")}')
print(f'  center_label: {npz.get("center_label", "none")}')

# Check X_m15 for 5x5 std
m15 = npz.get('X_m15')
if m15 is not None:
    print(f'  X_m15 shape: {m15.shape}, min/max: {m15.min():.2f} / {m15.max():.2f}')

# Now open the corresponding .nc
nc_path = nc_files[0]
print(f'\nNC file: {os.path.basename(nc_path)}')
ds = nc.Dataset(nc_path, 'r')
icm = ds['geophysical_data/Integer_Cloud_Mask'][:]
lat = ds['geolocation_data/latitude'][:]
lon = ds['geolocation_data/longitude'][:]
print(f'  Integer_Cloud_Mask: shape={icm.shape} dtype={icm.dtype}')
print(f'  ICM unique values: {np.unique(icm)}')
print(f'  ICM counts: {dict(zip(*np.unique(icm, return_counts=True)))}')
ds.close()

# Show mean CLDMSK classification in a few regions
print(f'\n  ICM at center (1616, 1600): {icm[1616, 1600]}')
print(f'  ICM at center (1600, 1616): {icm[1600, 1616]}')
print(f'  ICM in 5x5 around (1600, 1600):')
print(icm[1598:1603, 1598:1603])

# Cross-check: the npz Y_mask in center pixel
print(f'\n  npz Y_mask at center: {y_mask[64, 64]}')
print(f'  npz Y_mask 3x3 around center:')
print(y_mask[63:66, 63:66])
