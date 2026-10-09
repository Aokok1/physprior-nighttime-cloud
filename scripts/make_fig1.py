"""Figure 1: what the one-directional correction does, spatially and statistically.

Panel (a) shows one radar-clear Longmen test scene: the CLDMSK four-class field,
the M15 brightness temperature the rule tests, and the corrected field with the
flipped pixels outlined. Panel (b) puts all 197 radar-collocated test pixels in
the rule's own (sigma_5, M15) plane, so the operating point and the two flip
outcomes are visible directly.

Keeping the figure about the label field rather than about a network answers the
reviewer objection that nothing in the manuscript ever showed the corrected
labels themselves.

Run:  python -X utf8 scripts/make_fig1.py
Out:  E:/Claude code/paper/noise_label_grsl/figures/fig1_correction.pdf
"""
import csv
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from methods.physical_prior import apply_physical_correction  # noqa: E402

ROOT = r"E:/Claude code/project/noise-label-cloud"
DATASET = r"E:/Data/Unet_Dataset"
OUT = r"E:/Claude code/paper/noise_label_grsl/figures/fig1_correction.pdf"

SCENE = "Tensor_Longmen_A2020242.1724.npz"
T_TH, S_TH = 264.0, 1.5

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    # STIX is bundled with matplotlib, is Times-like (so it matches the IEEEtran
    # body text), and mathtext.fontset="stix" keeps the math in the same face.
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # IEEE PDF eXpress rejects Type 3 fonts. matplotlib's PDF backend defaults to
    # pdf.fonttype=3; 42 embeds TrueType instead. This is a submission blocker,
    # not a cosmetic setting.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})

# ------------------------------------------------------------------ panel (a)
z = np.load(os.path.join(DATASET, "Test", SCENE), allow_pickle=True)
ym = np.squeeze(z["Y_mask"]).astype(np.int32)
m15 = np.asarray(z["X_m15"], dtype=np.float32)
corr, _, _ = apply_physical_correction(ym, m15, m15_min=T_TH, std_max=S_TH)
flip = (ym >= 2) & (corr == 0)
ry, rx = (int(v) for v in z["Radar_Loc"])
truth = int(z["Center_Label"])
n_flip = int(flip.sum())

fig = plt.figure(figsize=(3.45, 2.72))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.02], hspace=0.28, wspace=0.10,
                      left=0.105, right=0.99, top=0.925, bottom=0.135)

cls_cmap = ListedColormap(["#cfe3f7", "#8fc0e8", "#f2b6a0", "#c9463c"])
cls_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cls_cmap.N)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(ym, cmap=cls_cmap, norm=cls_norm, interpolation="nearest")
ax.set_title("CLDMSK", pad=3)
ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(gs[0, 1])
lo, hi = np.nanpercentile(m15, [2, 98])
im = ax.imshow(m15, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
ax.set_title("M15 (K)", pad=3)
ax.set_xticks([]); ax.set_yticks([])
# Colorbar drawn inside the panel so it does not steal width from panel 3.
cax = ax.inset_axes([0.05, 0.05, 0.40, 0.045])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=5, width=0.3, length=1.2, pad=1.0)
cb.set_ticks([lo, (lo + hi) / 2, hi])
cb.ax.set_xticklabels([f"{lo:.0f}", "", f"{hi:.0f}"])
cb.outline.set_linewidth(0.3)

ax = fig.add_subplot(gs[0, 2])
shown = np.where(flip, 4, corr)
cmap2 = ListedColormap(["#cfe3f7", "#8fc0e8", "#f2b6a0", "#c9463c", "#2f7d32"])
norm2 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap2.N)
ax.imshow(shown, cmap=cmap2, norm=norm2, interpolation="nearest")
ax.set_title("corrected", pad=3)
ax.set_xticks([]); ax.set_yticks([])

for a in fig.axes[:3]:
    a.plot(rx, ry, marker="o", ms=3.2, mfc="none", mec="k", mew=0.7)
    a.add_patch(Rectangle((rx - 1.5, ry - 1.5), 3, 3, fill=False,
                          ec="k", lw=0.5, ls=":"))

# green swatch key for the third panel, drawn under that panel only
fig.text(0.99, 0.565, f"\u25a0 flipped (n={n_flip})",
         fontsize=6, color="#2f7d32", va="center", ha="right")

fig.text(0.01, 0.925, "(a)", fontsize=7.5)

# ------------------------------------------------------------------ panel (b)
man = [r for r in csv.DictReader(open(os.path.join(ROOT, "output", "manifest",
                                                   "sample_manifest_grp.csv"),
                                     encoding="utf-8", newline=""))
       if r["split"] == "test"]
sig = np.array([float(r["m15_local_std"]) for r in man])
t15 = np.array([float(r["m15_at_radar"]) for r in man])
ycls = np.array([int(float(r["y_mask_class_at_radar"])) for r in man])
rad = np.array([int(float(r["radar_label"])) for r in man])

eligible = ycls >= 2
fires = eligible & (t15 >= T_TH) & (sig <= S_TH)
benef = fires & (rad == 0)
over = fires & (rad == 1)
untouched = ~fires
kept = fires & (rad == 1) & (ycls == 3)

ax = fig.add_subplot(gs[1, :])
ax.axvline(T_TH, color="0.35", lw=0.6, ls="--")
ax.axhline(S_TH, color="0.35", lw=0.6, ls="--")
ax.scatter(t15[untouched], sig[untouched], s=5, c="0.72", lw=0, label="unchanged")
ax.scatter(t15[over], sig[over], s=12, c="#c9463c", lw=0, marker="v",
           label=f"over-correction (n={over.sum()})")
ax.scatter(t15[benef], sig[benef], s=12, c="#2f7d32", lw=0, marker="^",
           label=f"apparent FP corrected (n={benef.sum()})")
ax.set_xlabel("M15 brightness temperature at the radar pixel (K)")
ax.set_ylabel(r"local 5$\times$5 $\sigma$ (K)")
ax.text(T_TH + 0.6, ax.get_ylim()[1] * 0.93, r"$T_{\rm th}$", fontsize=6.5)
ax.text(ax.get_xlim()[0] + 0.8, S_TH + 0.06, r"$\sigma_{\rm th}$", fontsize=6.5)
ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.1,
          labelspacing=0.25, markerscale=1.3)
ax.tick_params(length=2)
fig.text(0.01, 0.50, "(b)", fontsize=7.5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print(f"wrote {OUT}")
print(f"  panel (a): {SCENE}  flips={n_flip}  radar={'cloud' if truth else 'clear'}")
print(f"  panel (b): n={len(man)}  fires={fires.sum()}  "
      f"beneficial={benef.sum()}  over-correction={over.sum()}")
