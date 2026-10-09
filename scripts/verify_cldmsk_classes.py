"""Verify CLDMSK 4-class mapping from .nc file.
Read Integer_Cloud_Mask, count classes, and see if 0/1=cloud and 2/3=clear.
"""
import netCDF4 as nc
import numpy as np

f = 'I:/global_dnb_cloud/cldmsk_nc_v3/CLDMSK_L2_VIIRS_SNPP.A2023001.0130.002.2026033140911.nc'
ds = nc.Dataset(f, 'r')

icm = ds['geophysical_data/Integer_Cloud_Mask'][:]
attrs = {ak: ds['geophysical_data/Integer_Cloud_Mask'].getncattr(ak) for ak in ds['geophysical_data/Integer_Cloud_Mask'].ncattrs()}
print('Integer_Cloud_Mask attribute DataDescription:')
print('  ', attrs.get('DataDescription', '<missing>'))
print()
print(f'shape={icm.shape}, dtype={icm.dtype}, valid_min/max={attrs.get("valid_min")}/{attrs.get("valid_max")}')
print()

# All unique values
vals, counts = np.unique(icm, return_counts=True)
total = icm.size
print(f'Total pixels: {total}')
for v, c in zip(vals, counts):
    print(f'  value={v}: count={c} ({c/total*100:.2f}%)')
print()

# Cross-check with Clear_Sky_Confidence: should be HIGH (closer to 1) for class=2/3 and LOW for class=0/1
csc = ds['geophysical_data/Clear_Sky_Confidence'][:]
print('Clear_Sky_Confidence per class (mean and median):')
for v in vals:
    mask = (icm == v)
    if mask.sum() > 0:
        csc_in = csc[mask]
        print(f'  class={v}: count={mask.sum()}, mean CSC={csc_in.mean():.3f}, median CSC={np.median(csc_in):.3f}')
ds.close()

# Now also look at Cloud_Mask (6 bytes/pixel) to confirm bit mapping
ds = nc.Dataset(f, 'r')
cm = ds['geophysical_data/Cloud_Mask'][:]
# bit 1 and bit 2 are 4-class
print('\n--- Cloud_Mask (byte_segment dim) ---')
for i in range(6):
    a = cm[i]
    print(f'  byte_segment={i}: unique values={np.unique(a)}')
ds.close()
