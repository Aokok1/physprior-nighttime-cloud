"""Visualize CLDMSK 4-class encoding vs DNB for 2024-06-21 Beijing.
Goal: visually verify which class values (0,1,2,3) are CLOUD and which are CLEAR.
Clouds in DNB look like low contrast, bright/dim, fuzzy texture.
"""
import numpy as np
import matplotlib.pyplot as plt

f = 'I:/global_dnb_cloud/paired_200px/beijing/Beijing_20240621_1716530.npz'
d = np.load(f)
dnb = d['dnb']
cldmsk = d['cldmsk']
lat = d['lat']
lon = d['lon']
print(f'DNB shape: {dnb.shape}, range: [{dnb.min():.2f}, {dnb.max():.2f}]')
print(f'CLDMSK shape: {cldmsk.shape}, dtype: {cldmsk.dtype}')
print(f'CLDMSK unique values: {np.unique(cldmsk, return_counts=True)}')
print(f'Lat range: [{lat.min():.2f}, {lat.max():.2f}]')
print(f'Lon range: [{lon.min():.2f}, {lon.max():.2f}]')
print(f'Source: {d["cldmsk_source"]}')

# Build figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1) DNB grayscale
im0 = axes[0,0].imshow(dnb, cmap='gray', origin='upper')
axes[0,0].set_title(f'DNB grayscale (Beijing 2024-06-21 17:16 UTC)\nDoy=173; data source: {d["cldmsk_source"][-20:]}', fontsize=10)
axes[0,0].set_xlabel('column')
axes[0,0].set_ylabel('row')
plt.colorbar(im0, ax=axes[0,0], fraction=0.046, label='DNB radiance (nW/cm²/sr)')

# 2) CLDMSK raw 4-class
cmap_discrete = plt.cm.tab10(np.linspace(0, 1, 4))
from matplotlib.colors import ListedColormap, BoundaryNorm
cldmsk_cmap = ListedColormap(['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])  # 0,1,2,3
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cldmsk_cmap.N)
im1 = axes[0,1].imshow(cldmsk, cmap=cldmsk_cmap, norm=norm, origin='upper')
axes[0,1].set_title('CLDMSK 4-class (raw values, 0–3)\nBlue=0, Green=1, Orange=2, Red=3', fontsize=10)
cbar1 = plt.colorbar(im1, ax=axes[0,1], fraction=0.046, ticks=[0,1,2,3])

# 3) Two hypothesized cloud masks: A (0/1=cloud) vs B (2/3=cloud)
cloud_A = (cldmsk <= 1).astype(int)  # 0/1=cloud
cloud_B = (cldmsk >= 2).astype(int)  # 2/3=cloud
mask_cmap = ListedColormap(['#cccccc', '#000000'])
norm_b = BoundaryNorm([-0.5, 0.5, 1.5], mask_cmap.N)
im2a = axes[0,2].imshow(cloud_A, cmap=mask_cmap, norm=norm_b, origin='upper')
axes[0,2].set_title(f'Hypothesis A: 0/1=cloud, 2/3=clear\nCloud fraction = {cloud_A.mean()*100:.1f}%', fontsize=10)
plt.colorbar(im2a, ax=axes[0,2], fraction=0.046, ticks=[0,1])

# 4) B
im2b = axes[1,0].imshow(cloud_B, cmap=mask_cmap, norm=norm_b, origin='upper')
axes[1,0].set_title(f'Hypothesis B: 2/3=cloud, 0/1=clear\nCloud fraction = {cloud_B.mean()*100:.1f}%', fontsize=10)
plt.colorbar(im2b, ax=axes[1,0], fraction=0.046, ticks=[0,1])

# 5) DNB with A-overlay
axes[1,1].imshow(dnb, cmap='gray', origin='upper')
axes[1,1].contour(cloud_A, levels=[0.5], colors='red', linewidths=0.5)
axes[1,1].set_title('DNB + Hyp A cloud edges (red)', fontsize=10)

# 6) DNB with B-overlay
axes[1,2].imshow(dnb, cmap='gray', origin='upper')
axes[1,2].contour(cloud_B, levels=[0.5], colors='red', linewidths=0.5)
axes[1,2].set_title('DNB + Hyp B cloud edges (red)', fontsize=10)

plt.suptitle('CLDMSK 4-class vs DNB ground truth — Beijing 2024-06-21 17:16 UTC', fontsize=13, fontweight='bold')
plt.tight_layout()
out_path = 'E:/Claude code/project/noise-label-cloud/output/visualization/cldmsk_4class_vs_dnb_beijing_20240621.png'
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f'Saved: {out_path}')

# Statistics: pixel-wise DNB mean/median within each CLDMSK class
print('\nDNB statistics by CLDMSK class:')
for v in np.unique(cldmsk):
    mask = (cldmsk == v)
    if mask.sum() > 0:
        dnb_in = dnb[mask]
        print(f'  class={v}: count={mask.sum()} ({mask.sum()/cldmsk.size*100:.1f}%) | DNB mean={dnb_in.mean():.2f} median={np.median(dnb_in):.2f} std={dnb_in.std():.2f}')

# Class co-occurrence with low-DNB (fuzzy) regions
# Low DNB std within 5x5 window = spatially fuzzy = could be cloud edge
from scipy.ndimage import uniform_filter
dnb_local_std = np.zeros_like(dnb)
for r in range(2, dnb.shape[0]-2):
    for c in range(2, dnb.shape[1]-2):
        dnb_local_std[r,c] = dnb[r-2:r+3, c-2:c+3].std()
print('\nDNB local std by CLDMSK class (5x5 window):')
for v in np.unique(cldmsk):
    mask = (cldmsk == v)
    if mask.sum() > 0:
        std_in = dnb_local_std[mask]
        # exclude edges
        valid = std_in[std_in > 0]
        if len(valid) > 0:
            print(f'  class={v}: DNB local std mean={valid.mean():.2f} median={np.median(valid):.2f}')
