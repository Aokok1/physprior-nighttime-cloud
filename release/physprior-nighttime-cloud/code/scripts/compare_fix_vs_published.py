"""Effect of the two data-side fixes on the published deep-learning rows.

Fix 1 (M15 frame alignment): the physical prior was evaluated on X_m15 read from
disk while the label mask had already been flipped/rotated by the loader.
Fix 2 (rotation bookkeeping): data/dataset.py _augment advanced ry/rx with the
clockwise rot90 index map while np.rot90 rotates counter-clockwise, so for k = 1
and k = 3 (half of all training views) radar_loc pointed at the wrong pixel of
the rotated patch and the centre-pixel supervision was mis-anchored.

Evaluation is untouched by both (the test and val loaders use augment=False), so
every row here is scored with the published 3x3-OR protocol on the same 197
radar-reference pixels.

Usage: python scripts/compare_fix_vs_published.py
"""
import json, csv, math

ROOT = r"E:/Claude code/project/noise-label-cloud"
WARM, STDMAX = 264.0, 1.5

pub = json.load(open(rf"{ROOT}/output/frozen_predictions/all_frozen_predictions_v2_grp.json"))
R = {k: {d["fname"]: (d["pred"], d["truth"]) for d in v} for k, v in pub.items()}

abl = json.load(open(rf"{ROOT}/output/2x2_ablation_fix.json"))
for c in abl:
    R[f"fix_abl_{c['label_mode']}_{c['n_channels']}ch"] = {
        d["fname"]: (d["pred"], d["truth"]) for d in c["test_records"]}
try:
    ctl = json.load(open(rf"{ROOT}/output/m15_label_control_fix.json"))
    for c in ctl:
        R[f"fix_ctl_{c['label_mode']}_{c['n_channels']}ch"] = {
            d["fname"]: (d["pred"], d["truth"]) for d in c["test_records"]}
except FileNotFoundError:
    ctl = []

M = {r["fname"]: r for r in csv.DictReader(open(rf"{ROOT}/output/manifest/sample_manifest_grp.csv"))}
test = sorted(R["CLDMSK_raw"])
cloud = [f for f in test if R["CLDMSK_raw"][f][1] == 1]
warm_c = [f for f in cloud if float(M[f]["m15_at_radar"]) >= WARM]
cold_c = [f for f in cloud if float(M[f]["m15_at_radar"]) < WARM]
clear = [f for f in test if R["CLDMSK_raw"][f][1] == 0]
cold_x = [f for f in clear if float(M[f]["m15_at_radar"]) < WARM]


def met(name, fs):
    tp = sum(1 for f in fs if R[name][f] == (1, 1)); fp = sum(1 for f in fs if R[name][f] == (1, 0))
    fn = sum(1 for f in fs if R[name][f] == (0, 1)); tn = sum(1 for f in fs if R[name][f] == (0, 0))
    n = len(fs)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(acc=(tp + tn) / n * 100, rec=tp / max(tp + fn, 1) * 100,
                spe=tn / max(tn + fp, 1) * 100, tp=tp,
                ba=(tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2 * 100,
                mcc=(tp * tn - fp * fn) / den if den else float("nan"))


def mcn(a, b, fs):
    x = sum(1 for f in fs if R[a][f][0] == R[a][f][1] and R[b][f][0] != R[b][f][1])
    y = sum(1 for f in fs if R[a][f][0] != R[a][f][1] and R[b][f][0] == R[b][f][1])
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


PAIRS = [
    ("raw labels + DL 2ch",        "abl_raw_2ch",         "fix_abl_raw_2ch"),
    ("raw labels + DL 3ch",        "abl_raw_3ch",         "fix_abl_raw_3ch"),
    ("PhysPrior labels + DL 2ch",  "abl_physprior_2ch",   "fix_abl_physprior_2ch"),
    ("PhysPrior labels + DL 3ch",  "abl_physprior_3ch",   "fix_abl_physprior_3ch"),
]
EXTRA = [(f"M15-threshold labels + DL {c}ch", "PureM15_T264", f"fix_ctl_m15thresh_{c}ch") for c in (2, 3)]
EXTRA += [(f"PhysPrior (aligned, true BT) + DL {c}ch", "abl_physprior_%dch" % c, f"fix_ctl_physprior_{c}ch") for c in (2, 3)]
EXTRA += [(f"PhysPrior (published path) + DL {c}ch", "abl_physprior_%dch" % c, f"fix_ctl_physprior_disk_{c}ch") for c in (2, 3)]

print("=== published vs fixed, on the same 197 radar-reference pixels ===")
print(f"{'row':34s}{'version':10s}{'acc':>8}{'BA':>8}{'recall':>8}{'spec':>7}{'MCC':>8}{'warm clouds':>13}")
for lab, old, new in PAIRS + EXTRA:
    for tag, key in (("published", old), ("fixed", new)):
        if key not in R:
            print(f"{lab:34s}{tag:10s}   [not run yet: {key}]")
            continue
        m = met(key, test); w = met(key, warm_c)
        print(f"{lab:34s}{tag:10s}{m['acc']:8.2f}{m['ba']:8.2f}{m['rec']:8.1f}{m['spe']:7.1f}"
              f"{m['mcc']:+8.3f}{w['tp']:8d}/{len(warm_c)}")
    if old in R and new in R:
        x, y, p = mcn(new, old, test)
        mo, mn = met(old, test), met(new, test)
        print(f"{'':34s}  delta acc {mn['acc']-mo['acc']:+.2f} pp, BA {mn['ba']-mo['ba']:+.2f} pp"
              f" | paired McNemar fixed-only={x} published-only={y} p={p:.4f}")
    print()

print("=== reference rows (not re-trained; the fixes cannot touch them) ===")
for lab in ("PureM15_T264", "PhysPriorLabel_moderate", "CLDMSK_raw", "Always_clear"):
    m = met(lab, test); w = met(lab, warm_c)
    print(f"  {lab:26s} acc {m['acc']:6.2f}  BA {m['ba']:6.2f}  recall {m['rec']:5.1f}  "
          f"spec {m['spe']:5.1f}  warm clouds {w['tp']}/{len(warm_c)}  cold-clear FP {met(lab, cold_x)['tp'] and 0}{''}")
