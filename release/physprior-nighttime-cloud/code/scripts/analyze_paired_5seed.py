"""Paired-seed read-out of the label-source block (stage 01 of rerun_fix.py).

30 runs = 3 label sources x 2 channel sets x 5 seeds, one fixed 20-epoch budget,
no early stopping, no checkpoint selection, scored on the same 197
radar-reference pixels with the published 3x3-OR protocol.

Because every cell shares the budget and the seed list, the seed is the only
thing that varies between sources: per-seed paired McNemar is exact, and the
spread over seeds is reported as an interval rather than as a single number
(a single seed moves by up to 4 pp here, which is why the earlier single-seed
comparisons could not be interpreted).
"""
import json, csv, math, statistics as st

ROOT = r"E:/Claude code/project/noise-label-cloud"
WARM = 264.0

runs = json.load(open(rf"{ROOT}/output/label_source_paired_5seed_fix.json"))
M = {r["fname"]: r for r in csv.DictReader(open(rf"{ROOT}/output/manifest/sample_manifest_grp.csv"))}
pub = json.load(open(rf"{ROOT}/output/frozen_predictions/all_frozen_predictions_v2_grp.json"))
truth = {d["fname"]: d["truth"] for d in pub["CLDMSK_raw"]}
test = sorted(truth)
cloud = [f for f in test if truth[f] == 1]
warm_c = [f for f in cloud if float(M[f]["m15_at_radar"]) >= WARM]

R = {}
for r in runs:
    R[(r["label_mode"], r["n_channels"], r["seed"])] = {
        d["fname"]: (d["pred"], d["truth"]) for d in r["test_records"]}
# the two direct rules act on labels, not on trained weights: no seed variance
for k in ("PureM15_T264", "PhysPriorLabel_moderate", "CLDMSK_raw"):
    R[k] = {d["fname"]: (d["pred"], d["truth"]) for d in pub[k]}


def met(rec, fs):
    tp = sum(1 for f in fs if rec[f] == (1, 1)); fp = sum(1 for f in fs if rec[f] == (1, 0))
    fn = sum(1 for f in fs if rec[f] == (0, 1)); tn = sum(1 for f in fs if rec[f] == (0, 0))
    n = len(fs)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    rec_pc = tp / max(tp + fn, 1) * 100; spe = tn / max(tn + fp, 1) * 100
    return dict(acc=(tp + tn) / n * 100, ba=(rec_pc + spe) / 2, recall=rec_pc, spec=spe,
                mcc=(tp * tn - fp * fn) / den if den else float("nan"), warm_tp=tp)


def mcn(a, b, fs):
    x = sum(1 for f in fs if a[f][0] == a[f][1] and b[f][0] != b[f][1])
    y = sum(1 for f in fs if a[f][0] != a[f][1] and b[f][0] == b[f][1])
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


SEEDS = sorted({k[2] for k in R if isinstance(k, tuple)})
print(f"=== 1. per-cell five-seed summary (mean +/- sample sd over seeds {SEEDS}) ===")
hdr = f"{'label source / channels':34s}{'acc':>16}{'BA':>16}{'cloud recall':>18}{'warm clouds found':>22}"
print(hdr)
for mode in ("raw", "physprior", "m15thresh"):
    for ch in (2, 3):
        keys = [(mode, ch, s) for s in SEEDS if (mode, ch, s) in R]
        if not keys:
            continue
        def col(name):
            v = [met(R[k], test)[name] for k in keys]
            return st.mean(v), st.stdev(v) if len(v) > 1 else 0.0
        a, asd = col("acc"); b, bsd = col("ba"); c, csd = col("recall")
        w = [met(R[k], warm_c)["warm_tp"] for k in keys]
        acc_all = [met(R[k], test)["acc"] for k in keys]
        print(f"{mode + ' + DL ' + str(ch) + 'ch':34s}"
              f"{a:9.2f} ±{asd:5.2f}  {b:8.2f} ±{bsd:4.2f}  {c:9.1f} ±{csd:4.1f}"
              f"   {st.mean(w):5.1f} ±{st.stdev(w):4.1f} of {len(warm_c)}   "
              f"[{min(acc_all):.2f}–{max(acc_all):.2f}]")

print("\n=== 2. direct rules (no training, no seed variance) ===")
for k, lab in (("CLDMSK_raw", "raw CLDMSK label"), ("PhysPriorLabel_moderate", "PhysPrior label"),
               ("PureM15_T264", "M15 threshold label")):
    m = met(R[k], test); w = met(R[k], warm_c)
    print(f"  {lab:24s} acc {m['acc']:6.2f}  BA {m['ba']:6.2f}  recall {m['recall']:5.1f}  "
          f"spec {m['spec']:5.1f}  MCC {m['mcc']:+.3f}  warm clouds {w['warm_tp']}/{len(warm_c)}")

print("\n=== 3. per-seed paired McNemar (same seed, same budget, only the label source differs) ===")
def block(a_mode, b_mode, label):
    print(f"\n  {label}")
    for ch in (2, 3):
        rows = []
        for s in SEEDS:
            ka, kb = (a_mode, ch, s), (b_mode, ch, s)
            if ka in R and kb in R:
                x, y, p = mcn(R[ka], R[kb], test)
                xw, yw, pw = mcn(R[ka], R[kb], cloud)
                rows.append((s, x, y, p, xw, yw, pw))
        if not rows:
            continue
        wins = sum(1 for r in rows if r[1] > r[2])
        mean_d = st.mean([met(R[(a_mode, ch, r[0])], test)["acc"] -
                          met(R[(b_mode, ch, r[0])], test)["acc"] for r in rows])
        sd_d = st.stdev([met(R[(a_mode, ch, r[0])], test)["acc"] -
                         met(R[(b_mode, ch, r[0])], test)["acc"] for r in rows]) if len(rows) > 1 else 0
        print(f"    {ch} ch: {a_mode} wins the seed in {wins}/{len(rows)} | mean acc delta {mean_d:+.2f} ±{sd_d:.2f} pp")
        for s, x, y, p, xw, yw, pw in rows:
            print(f"      seed {s}: overall {x:2d}-{y:2d} p={p:.4f} | cloudy pixels {xw:2d}-{yw:2d} p={pw:.4f}")

block("physprior", "raw", "PhysPrior labels vs raw CLDMSK labels  (the paper's headline claim)")
block("m15thresh", "physprior", "M15-threshold labels vs PhysPrior labels")

print("\n=== 4. combined across seeds: how often is each label source strictly dominated? ===")
for ch in (2, 3):
    for a, b in (("physprior", "raw"), ("m15thresh", "physprior"), ("m15thresh", "raw")):
        sig = 0; tot = 0
        for s in SEEDS:
            ka, kb = (a, ch, s), (b, ch, s)
            if ka in R and kb in R:
                x, y, p = mcn(R[ka], R[kb], cloud); tot += 1
                sig += int(p < 0.05)
        print(f"  {ch} ch {a:>10s} vs {b:<10s} on the 58 cloudy pixels: p<0.05 in {sig}/{tot} seeds")
