"""Stratified read-out of the M15-threshold label control (run after
scripts/train_m15_label_control.py).

Every number is recomputed from per-sample prediction records: the control
cells from output/m15_label_control.json, the published rows from
output/frozen_predictions/all_frozen_predictions_v2_grp.json, and the physical
strata from the m15_at_radar / m15_local_std / y_mask_class_at_radar columns of
output/manifest/sample_manifest_grp.csv.

The four strata are the whole point: a single 11-um threshold cannot label a
cloud whose radar-pixel brightness temperature is >= 264 K, because that
constant *is* its decision boundary. PhysPrior keeps part of those pixels
because it only edits CLDMSK's own cloudy classes and requires local uniformity
as well.
"""
import json, csv, math, sys

ROOT = r"E:/Claude code/project/noise-label-cloud"
WARM_T = 264.0
STD_MAX = 1.5

ctl = json.load(open(rf"{ROOT}/output/m15_label_control.json"))
P = json.load(open(rf"{ROOT}/output/frozen_predictions/all_frozen_predictions_v2_grp.json"))
M = {r["fname"]: r for r in csv.DictReader(open(rf"{ROOT}/output/manifest/sample_manifest_grp.csv"))}

R = {k: {d["fname"]: (d["pred"], d["truth"]) for d in v} for k, v in P.items()}
for c in ctl:
    R[f"ctl_{c['label_mode']}_{c['n_channels']}ch"] = {d["fname"]: (d["pred"], d["truth"])
                                                       for d in c["test_records"]}
test = sorted(R["CLDMSK_raw"])
warm_c = [f for f in test if R["CLDMSK_raw"][f][1] == 1 and float(M[f]["m15_at_radar"]) >= WARM_T]
cold_c = [f for f in test if R["CLDMSK_raw"][f][1] == 1 and float(M[f]["m15_at_radar"]) < WARM_T]
warm_x = [f for f in test if R["CLDMSK_raw"][f][1] == 0 and float(M[f]["m15_at_radar"]) >= WARM_T]
cold_x = [f for f in test if R["CLDMSK_raw"][f][1] == 0 and float(M[f]["m15_at_radar"]) < WARM_T]
cloud = warm_c + cold_c
clear = warm_x + cold_x


def met(name, fs):
    tp = sum(1 for f in fs if R[name][f] == (1, 1))
    fp = sum(1 for f in fs if R[name][f] == (1, 0))
    fn = sum(1 for f in fs if R[name][f] == (0, 1))
    tn = sum(1 for f in fs if R[name][f] == (0, 0))
    n = len(fs)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn, acc=(tp + tn) / n * 100 if n else float("nan"),
                rec=tp / (tp + fn) * 100 if tp + fn else float("nan"),
                spe=tn / (tn + fp) * 100 if tn + fp else float("nan"),
                ba=(tp / (tp + fn) + tn / (tn + fp)) / 2 * 100 if tp + fn and tn + fp else float("nan"),
                mcc=(tp * tn - fp * fn) / den if den else float("nan"))


def mcnemar(a, b, fs):
    x = sum(1 for f in fs if R[a][f][0] == R[a][f][1] and R[b][f][0] != R[b][f][1])
    y = sum(1 for f in fs if R[a][f][0] != R[a][f][1] and R[b][f][0] == R[b][f][1])
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


ROWS = [("PureM15_T264 (published row)", "PureM15_T264"),
        ("PhysPrior label (direct)", "PhysPriorLabel_moderate"),
        ("CLDMSK raw label", "CLDMSK_raw"),
        ("abl: raw labels + DL 2ch", "abl_raw_2ch"),
        ("abl: raw labels + DL 3ch", "abl_raw_3ch"),
        ("abl: PhysPrior + DL 2ch (published, disk M15)", "abl_physprior_2ch"),
        ("abl: PhysPrior + DL 3ch (published, disk M15)", "abl_physprior_3ch"),
        ("ctl: raw labels + DL 2ch (reproduction)", "ctl_raw_2ch"),
        ("ctl: raw labels + DL 3ch (reproduction)", "ctl_raw_3ch"),
        ("ctl: PhysPrior + DL 2ch, M15 read from disk", "ctl_physprior_disk_2ch"),
        ("ctl: PhysPrior + DL 2ch, labels frame-aligned", "ctl_physprior_2ch"),
        ("ctl: PhysPrior + DL 3ch, labels frame-aligned", "ctl_physprior_3ch"),
        ("ctl: M15-threshold LABELS + DL 2ch", "ctl_m15thresh_2ch"),
        ("ctl: M15-threshold LABELS + DL 3ch", "ctl_m15thresh_3ch")]

have = {n for _, n in ROWS if n in R}
print("=== 1. overall on the 197 radar-reference pixels ===")
print(f"{'row':52s}{'acc':>7}{'BA':>7}{'rec':>7}{'spec':>7}{'MCC':>8}  TN/FP/FN/TP")
for lab, n in ROWS:
    if n not in R:
        print(f"{lab:52s}   [not run yet]")
        continue
    m = met(n, test)
    print(f"{lab:52s}{m['acc']:7.2f}{m['ba']:7.2f}{m['rec']:7.1f}{m['spe']:7.1f}{m['mcc']:+8.3f}"
          f"  {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}")

print("\n=== 2. the stratum that decides the argument: clouds with M15 >= 264 K (n=%d) ===" % len(warm_c))
print(f"{'row':52s}{'warm clouds found':>20}{'cold clouds':>13}{'FP warm-clear':>15}{'FP cold-clear':>15}")
for lab, n in ROWS:
    if n not in R:
        continue
    a, b, c, d = met(n, warm_c)["tp"], met(n, cold_c)["tp"], met(n, warm_x)["fp"], met(n, cold_x)["fp"]
    print(f"{lab:52s}{a:>10d}/{len(warm_c)}{b:>10d}/{len(cold_c)}{c:>12d}/{len(warm_x)}{d:>12d}/{len(cold_x)}")

print("\n=== 3. paired exact McNemar vs PureM15_T264 ===")
print(f"{'row':52s}{'this-only':>10}{'M15-only':>10}{'p':>10}   (within the 58 cloudy pixels)")
for lab, n in ROWS:
    if n not in R or n == "PureM15_T264":
        continue
    x, y, p = mcnemar(n, "PureM15_T264", cloud)
    print(f"{lab:52s}{x:>10d}{y:>10d}{p:>10.5f}")

print("\n=== 4. head-to-head: M15-threshold labels vs PhysPrior labels (same budget) ===")
for ch in (2, 3):
    a, b = f"ctl_m15thresh_{ch}ch", f"ctl_physprior_{ch}ch"
    if a in R and b in R:
        x, y, p = mcnemar(b, a, test)
        xa, ya, pa = mcnemar(b, a, cloud)
        print(f"  {ch} ch: overall PhysPrior-only={x} M15label-only={y} p={p:.4f} | "
              f"cloudy pixels: PP-only={xa} M15label-only={ya} p={pa:.4f}")
        for lab, n in [(f"M15labels {ch}ch", a), (f"PhysPrior {ch}ch", b)]:
            m = met(n, test)
            print(f"     {lab:20s} acc {m['acc']:6.2f}  BA {m['ba']:6.2f}  rec {m['rec']:5.1f}  "
                  f"warm-cloud TP {met(n, warm_c)['tp']}/{len(warm_c)}")

print("\n=== 5. verdict keys ===")
need = ["ctl_m15thresh_2ch", "ctl_m15thresh_3ch", "ctl_physprior_2ch", "ctl_physprior_3ch"]
missing = [n for n in need if n not in R]
if missing:
    print("  control cells still missing:", ", ".join(missing))
else:
    a = max(met(f"ctl_m15thresh_{c}ch", test)["acc"] for c in (2, 3))
    b = max(met(f"ctl_physprior_{c}ch", test)["acc"] for c in (2, 3))
    aw, bw = max(met(f"ctl_m15thresh_{c}ch", warm_c)["tp"] for c in (2, 3)), \
             max(met(f"ctl_physprior_{c}ch", warm_c)["tp"] for c in (2, 3))
    print(f"  best accuracy: M15-label {a:.2f} vs PhysPrior-label {b:.2f} -> "
          f"{'M15 labels are NOT beaten; rewrite the contribution as label-bias analysis' if a >= b else 'PhysPrior labels win on accuracy'}")
    print(f"  warm-top clouds detected: M15-label {aw}/{len(warm_c)} vs PhysPrior-label {bw}/{len(warm_c)}")
