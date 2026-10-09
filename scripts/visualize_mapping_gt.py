"""Visualize CLDMSK 2/3=cloud mapping vs DNB for Beijing 2024-06-21.
Follow the mapping convention used in gen_figz.py:
  cloud_mask = (cl == 2) | (cl == 3)
  clear_mask = (cl == 0) | (cl == 1)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

f = 'I:/global_dnb_cloud/paired_200px/beijing/Beijing_20230104_1912414.npz'
d = np.load(f, allow_pickle=True)
dn = d['dnb'].astype(np.float32)
cl = d['cldmsk'].astype(np.int8)
lat = d['lat']
lon = d['lon']
src = str(d['cldmsk_source'])

# Use the project convention: 2/3=cloud
cloud_mask = (cl == 2) | (cl == 3)
clear_mask = (cl == 0) | (cl == 1)
print(f'DNB range: [{dn.min():.4f}, {dn.max():.4f}]')
print(f'CLDMSK unique: {np.unique(cl, return_counts=True)}')
print(f'Cloud (2/3) fraction: {cloud_mask.mean()*100:.1f}%')
print(f'Clear (0/1) fraction: {clear_mask.mean()*100:.1f}%')

# Print class-wise DNB statistics
print('\nClass-wise DNB statistics:')
for v in [0, 1, 2, 3]:
    mask = (cl == v)
    if mask.sum() > 0:
        dnb_in = dn[mask]
        print(f'  class={v}: count={mask.sum():5d} ({mask.sum()/cl.size*100:5.1f}%) '
              f'| DNB mean={dnb_in.mean():.3f} median={np.median(dnb_in):.3f} std={dnb_in.std():.3f} '
              f'| min={dnb_in.min():.3f} max={dnb_in.max():.3f}')

# DNB local std (5x5) as cloud-edge indicator
from scipy.ndimage import uniform_filter
dnb_smooth = uniform_filter(dn, size=5)
dnb_local_var = uniform_filter(dn**2, size=5) - dnb_smooth**2
dnb_local_std = np.sqrt(np.clip(dnb_local_var, 0, None))

print('\nClass-wise DNB local std (5x5):')
for v in [0, 1, 2, 3]:
    mask = (cl == v)
    if mask.sum() > 0:
        std_in = dnb_local_std[mask]
        valid = std_in[std_in > 0]
        if len(valid) > 0:
            print(f'  class={v}: DNB 5x5 std mean={valid.mean():.3f} median={np.median(valid):.3f}')

# Build figure
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.25,
              left=0.06, right=0.98, top=0.92, bottom=0.06)

# Log-stretched DNB
d_log = np.log10(np.clip(dn, 1e-10, None) + 1e-10)
d_show = (d_log - d_log.min()) / (d_log.max() - d_log.min() + 1e-8)

# 1) DNB
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(d_show, cmap='inferno', vmin=0, vmax=1, aspect='equal', origin='lower')
ax0.set_title('DNB radiance (Inferno, log-scaled)', fontsize=11, fontweight='bold')
ax0.set_xticks(np.arange(0, 201, 50)); ax0.set_yticks(np.arange(0, 201, 50))
ax0.set_xlabel('Pixel x'); ax0.set_ylabel('Pixel y')

# 2) DNB grayscale
ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(dn, cmap='gray', aspect='equal', origin='lower')
ax1.set_title('DNB grayscale (raw)', fontsize=11, fontweight='bold')
ax1.set_xticks(np.arange(0, 201, 50)); ax1.set_yticks(np.arange(0, 201, 50))
ax1.set_xlabel('Pixel x'); ax1.set_ylabel('Pixel y')

# 3) CLDMSK 4-class (raw values)
ax2 = fig.add_subplot(gs[0, 2])
im2 = ax2.imshow(cl, cmap='viridis', vmin=0, vmax=3, aspect='equal', origin='lower')
ax2.set_title('CLDMSK raw 4-class (0,1,2,3)\nProject mapping: 2/3=cloud', fontsize=11, fontweight='bold')
ax2.set_xticks(np.arange(0, 201, 50)); ax2.set_yticks(np.arange(0, 201, 50))
ax2.set_xlabel('Pixel x'); ax2.set_ylabel('Pixel y')
plt.colorbar(im2, ax=ax2, fraction=0.046, ticks=[0,1,2,3])

# 4) Cloud mask: 0/1=clear (black), 2/3=cloud (white)
ax3 = fig.add_subplot(gs[0, 3])
cldmsk_disp = 1 - cloud_mask.astype(float)
ax3.imshow(cldmsk_disp, cmap='gray', vmin=0, vmax=1, aspect='equal', origin='lower')
ax3.set_title(f'Project cloud mask (2/3=cloud, white)\nCloud fraction = {cloud_mask.mean()*100:.1f}%', fontsize=11, fontweight='bold')
ax3.set_xticks(np.arange(0, 201, 50)); ax3.set_yticks(np.arange(0, 201, 50))
ax3.set_xlabel('Pixel x'); ax3.set_ylabel('Pixel y')

# 5) DNB + 2/3 overlay
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(d_show, cmap='inferno', vmin=0, vmax=1, aspect='equal', origin='lower')
ax4.contour(cloud_mask, levels=[0.5], colors='cyan', linewidths=0.7)
ax4.set_title('DNB + 2/3=cloud overlay (cyan contour)\nThe mapping that gen_figz.py uses', fontsize=11, fontweight='bold')
ax4.set_xticks(np.arange(0, 201, 50)); ax4.set_yticks(np.arange(0, 201, 50))
ax4.set_xlabel('Pixel x'); ax4.set_ylabel('Pixel y')

# 6) DNB + 0/1 overlay (NASA official)
ax5 = fig.add_subplot(gs[1, 1])
nasa_cloud = (cl == 0) | (cl == 1)
ax5.imshow(d_show, cmap='inferno', vmin=0, vmax=1, aspect='equal', origin='lower')
ax5.contour(nasa_cloud, levels=[0.5], colors='lime', linewidths=0.7)
ax5.set_title(f'DNB + NASA 0/1=cloud overlay (lime contour)\nCloud fraction = {nasa_cloud.mean()*100:.1f}%', fontsize=11, fontweight='bold')
ax5.set_xticks(np.arange(0, 201, 50)); ax5.set_yticks(np.arange(0, 201, 50))
ax5.set_xlabel('Pixel x'); ax5.set_ylabel('Pixel y')

# 7) DNB local std
ax6 = fig.add_subplot(gs[1, 2])
im6 = ax6.imshow(dnb_local_std, cmap='magma', aspect='equal', origin='lower')
ax6.set_title('DNB local std (5x5) — texture\nClouds should have high local std (fuzzy)', fontsize=11, fontweight='bold')
ax6.set_xticks(np.arange(0, 201, 50)); ax6.set_yticks(np.arange(0, 201, 50))
ax6.set_xlabel('Pixel x'); ax6.set_ylabel('Pixel y')
plt.colorbar(im6, ax=ax6, fraction=0.046)

# 8) DNB + 2/3 cloud edges highlighted
ax7 = fig.add_subplot(gs[1, 3])
ax7.imshow(dnb_local_std, cmap='magma', aspect='equal', origin='lower')
ax7.contour(cloud_mask, levels=[0.5], colors='cyan', linewidths=0.7)
ax7.set_title('DNB local std + 2/3=cloud contour\nWhere cyan is on dark regions → cloud', fontsize=11, fontweight='bold')
ax7.set_xticks(np.arange(0, 201, 50)); ax7.set_yticks(np.arange(0, 201, 50))
ax7.set_xlabel('Pixel x'); ax7.set_ylabel('Pixel y')

import os
os.makedirs('E:/Claude code/project/noise-label-cloud/output/visualization', exist_ok=True)
out_path = 'E:/Claude code/project/noise-label-cloud/output/visualization/cldmsk_mapping_ground_truth.png'
plt.suptitle(f'Beijing 2024-06-21 17:16 UTC — Doy=173 — {src}',
             fontsize=12, fontweight='bold')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f'\nSaved: {out_path}')
