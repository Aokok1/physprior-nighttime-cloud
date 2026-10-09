"""Find ground truth: which class is cloud?
CSC = Clear Sky Confidence: high CSC = clear, low CSC = cloudy.
We expect:
  class 0 (cloudy) → low CSC
  class 3 (confident clear) → high CSC
"""
import netCDF4 as nc, numpy as np

f = 'I:/global_dnb_cloud/cldmsk_nc_v3/CLDMSK_L2_VIIRS_SNPP.A2024173.2042.001.2024174091000.nc'
ds = nc.Dataset(f, 'r')

# Just check attrs first
icm_attrs = {ak: ds['geophysical_data/Integer_Cloud_Mask'].getncattr(ak) for ak in ds['geophysical_data/Integer_Cloud_Mask'].ncattrs()}
print('Integer_Cloud_Mask attrs:')
for k, v in icm_attrs.items():
    print(f'  {k} = {v}')
print()

csc_attrs = {ak: ds['geophysical_data/Clear_Sky_Confidence'].getncattr(ak) for ak in ds['geophysical_data/Clear_Sky_Confidence'].ncattrs()}
print('Clear_Sky_Confidence attrs:')
for k, v in csc_attrs.items():
    print(f'  {k} = {v}')

icm = ds['geophysical_data/Integer_Cloud_Mask'][:]
csc = ds['geophysical_data/Clear_Sky_Confidence'][:]

print(f'\nicm shape: {icm.shape}, csc shape: {csc.shape}')
print()

# Masked array issue: print unmasked
icm_arr = np.ma.filled(icm, fill_value=-1)
csc_arr = np.ma.filled(csc, fill_value=-1)
print(f'icm unique (with -1): {np.unique(icm_arr, return_counts=True)}')

# Per-class CSC mean
for v in [0, 1, 2, 3, -1]:
    mask = (icm_arr == v)
    if mask.sum() > 0:
        csc_in = csc_arr[mask]
        csc_in_valid = csc_in[csc_in >= 0]
        if len(csc_in_valid) > 0:
            print(f'  class={v}: count={mask.sum()}, mean CSC={csc_in_valid.mean():.3f}, median={np.median(csc_in_valid):.3f}')

ds.close()
