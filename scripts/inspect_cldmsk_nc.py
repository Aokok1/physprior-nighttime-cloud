"""Check CLDMSK groups with safe attribute read."""
import netCDF4 as nc
f = 'I:/global_dnb_cloud/cldmsk_nc_v3/CLDMSK_L2_VIIRS_SNPP.A2023001.0130.002.2026033140911.nc'
ds = nc.Dataset(f, 'r')
def safe_attrs(v):
    return {ak: v.getncattr(ak) for ak in v.ncattrs()}

print('GROUPS:')
for gname, g in ds.groups.items():
    print(f'\n  === GROUP {gname} ===')
    for vn, v in g.variables.items():
        print(f'    {vn}: dtype={v.dtype} shape={v.shape} dims={v.dimensions}')
        for ak, av in safe_attrs(v).items():
            print(f'        {ak} = {str(av)[:90]}')
print('\n---TOP-LEVEL VARS---')
for vn, v in ds.variables.items():
    print(f'  {vn}: dtype={v.dtype} shape={v.shape}')
ds.close()
