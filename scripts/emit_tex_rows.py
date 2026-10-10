"""Emit the LaTeX rows the corrected manuscript needs, straight from the artifacts.

Nothing here is typed by hand: every percentage, interval and count is read from
output/paper_tables_grp.json, output/derived_tables_grp.json and
output/label_source_paired_5seed_grp.json, which are the promoted products of the
2026-10-10 corrected re-run. Five-seed rows use the population sd (ddof = 0) over
the five seeds, the convention the manuscript already prints (70.66 ± 3.60 was the
4.02 sample sd times sqrt(4/5)).

Usage: python scripts/emit_tex_rows.py
"""
import json, csv, math, statistics as st

ROOT = r"E:/Claude code/project/noise-label-cloud"
T = json.load(open(rf"{ROOT}/output/paper_tables_grp.json"))
D = json.load(open(rf"{ROOT}/output/derived_tables_grp.json"))
paired = json.load(open(rf"{ROOT}/output/label_source_paired_5seed_grp.json"))
FZ = json.load(open(rf"{ROOT}/output/frozen_predictions/all_frozen_predictions_v2_grp.json"))
M = {r["fname"]: r for r in csv.DictReader(open(rf"{ROOT}/output/manifest/sample_manifest_grp.csv"))}
POP = lambda v: st.stdev(v) * math.sqrt((len(v) - 1) / len(v))

t1 = T["table1"]


def row1(key, label):
    m = t1[key]["metrics"]; ci = m["ci"]
    return (f"{label} & 197 & {m['acc']*100:.1f}\\%\\,[{ci['lo']*100:.1f},\\,{ci['hi']*100:.1f}] & "
            f"{m['balanced_accuracy']*100:.1f} & {m['f1']*100:.1f} & {m['mcc']:+.3f} \\\\")


print("=== Table I rows that changed (paste over the existing lines) ===")
for key, lab in [("Standard_training", "Standard training"),
                 ("LossCorrection", "Loss Correction"),
                 ("CoTeaching_A", "Co-Teaching A"),
                 ("CoTeaching_B", "Co-Teaching B"),
                 ("GCE_q07", "GCE ($q{=}0.7$)"),
                 ("Mixup", "Mixup"),
                 ("PhysPrior_DL_3ch", "PhysPrior moderate + DL (3 channels)"),
                 ("PhysPrior_CoTeaching_ensemble", r"PhysPrior + Co-Teaching (predef.\ ensemble)")]:
    print(row1(key, lab) + ("   \\textbf{...}" if key == "PhysPrior_CoTeaching_ensemble" else ""))

print("\n=== Table III per-station rows ===")
ps = D["per_station"]
for k, name in (("CS", r"Changsha (urban)"), ("LM", r"Longmen (suburban)")):
    v = ps[k]
    print(f"{name} & {v['n']}  & {v['acc']*100:.1f}\\% & {v['precision']*100:.1f}\\% & "
          f"{v['recall']*100:.1f}\\% & {v['specificity']*100:.1f}\\% \\\\")
m = t1["PhysPrior_DL_3ch"]["metrics"]
print(f"Combined & 197 & {m['acc']*100:.1f}\\% & {m['precision']*100:.1f}\\% & {m['recall']*100:.1f}\\% & "
      f"{m['specificity']*100:.1f}\\% \\\\\"   (from Table I row: precision/recall/specificity)")

print("\n=== Table IV per-season rows (bootstrap 95% intervals from the artifact) ===")
for s in ("Spring", "Summer", "Autumn", "Winter"):
    v = D["per_season"][s]
    ci = v.get("ci", {}) or {}
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None:
        lo, hi = v.get("acc_ci", [None, None])[0], v.get("acc_ci", [None, None])[1]
    txt = f"[{lo*100:.1f}, {hi*100:.1f}]" if lo is not None else "[CI keys: %s]" % list(v)
    print(f"{s} ({v.get('months','')}) & {v['acc']*100:.1f}\\% & {txt} & {v['n']} \\\\")

print("\n=== Table V: label source x channels, five paired seeds, fixed 20-epoch budget ===")
R = {(r["label_mode"], r["n_channels"], r["seed"]): {d["fname"]: (d["pred"], d["truth"]) for d in r["test_records"]}
     for r in paired}
test = sorted(R[("raw", 2, 0)])
seeds = sorted({k[2] for k in R})
accs = {}
for mode in ("raw", "physprior", "m15thresh"):
    for ch in (2, 3):
        a = [sum(1 for f in test if R[(mode, ch, s)][f][0] == R[(mode, ch, s)][f][1]) / len(test) * 100
             for s in seeds]
        accs[(mode, ch)] = a
        print(f"{mode + ' labels':26s} & {a[0]:.1f}\\%,{a[1]:.1f}\\%,{a[2]:.1f}\\%,{a[3]:.1f}\\%,{a[4]:.1f}\\% & "
              f"{st.mean(a):.1f}\\% \\,$\\pm$\\, {POP(a):.2f}~pp & {min(a):.1f}--{max(a):.1f} \\%  "
              f"(mean over seeds 0--4)  [{ch} ch]")
for ch in (2, 3):
    d = [accs[("physprior", ch)][i] - accs[("raw", ch)][i] for i in range(len(seeds))]
    print(f"label effect {ch} ch: {st.mean(d):+.2f} ± {POP(d):.2f} pp  (per seed: "
          + ", ".join(f"{v:+.2f}" for v in d) + f"), {sum(1 for v in d if v > 0)}/5 positive")
for mode in ("raw", "physprior"):
    d = [accs[(mode, 3)][i] - accs[(mode, 2)][i] for i in range(len(seeds))]
    print(f"basemap effect under {mode}: {st.mean(d):+.2f} ± {POP(d):.2f} pp, {sum(1 for v in d if v < 0)}/5 negative")
d = [accs[("m15thresh", ch)][i] - accs[("physprior", ch)][i] for ch, i in [(2, 0)] for _ in [0]]
acc_delta = st.mean([accs[("m15thresh", 2)][i] - accs[("physprior", 2)][i] for i in range(5)])
acc_delta3 = st.mean([accs[("m15thresh", 3)][i] - accs[("physprior", 3)][i] for i in range(5)])
print(f"threshold-label minus corrected-label: {acc_delta:+.2f} pp (2 ch), {acc_delta3:+.2f} pp (3 ch)")

print("\n=== warm-top cloud detection per label source (of 26) ===")
tr = {d["fname"]: d["truth"] for d in FZ["CLDMSK_raw"]}
warm = [f for f in test if tr[f] == 1 and float(M[f]["m15_at_radar"]) >= 264.0]
for mode in ("raw", "physprior", "m15thresh"):
    for ch in (2, 3):
        w = [sum(1 for f in warm if R[(mode, ch, s)][f][0] == 1) for s in seeds]
        print(f"  {mode:10s} {ch} ch: {st.mean(w):.1f} ± {POP(w):.1f} of {len(warm)}  (per seed {w})")

print("\n=== Table VI: agreement + five-seed block ===")
A = D["agreement"]
n = A["n"]
print(f"Direct PhysPrior rule correct against radar         & {A['rule_correct']}/{n} ({A['rule_correct']/n*100:.2f}\\%) \\\\")
print(f"Downstream DL correct against radar                 & {A['model_correct']}/{n} ({A['model_correct']/n*100:.2f}\\%) \\\\")
print(f"\\emph{{Both}} correct against radar                   & {A['both_correct']}/{n} ({A['both_correct']/n*100:.2f}\\%) \\\\")
print(f"Direct PhysPrior only correct                       & {A['rule_only_correct']}/{n} \\\\")
print(f"Downstream DL only correct                          & {A['dl_only_correct']}/{n} \\\\")
print(f"Both incorrect                                      & {A['both_wrong']}/{n} \\\\")
print(f"Sample-wise agreement (label $=$ model)             & {A['agree']}/{n} ({A['agree']/n*100:.2f}\\%) \\\\")
print(f"Cohen's $\\kappa$                                   & ${A['kappa']:+.3f}$ \\\\")
print(f"McNemar discordant counts $(b,c)$                   & $({A['mcnemar_b']},\\,{A['mcnemar_c']})$ \\\\")
print(f"Exact McNemar $p$-value (two-sided)                 & ${A['mcnemar_p']:.3f}$ \\\\")
fp = T["table_five_seed_physprior"]["per_seed"]
fc = T["table_five_seed_coteaching"]["per_seed"]
for nm, key in (("PhysPrior+DL", fp), ("PhysPrior+Co-Teaching", fc)):
    a = [r["acc"] * 100 for r in key]; ba = [r["balanced_accuracy"] * 100 for r in key]
    f1 = [r["f1"] * 100 for r in key]; rc = [r["recall"] * 100 for r in key]; sp = [r["specificity"] * 100 for r in key]
    print(f"Five-seed mean accuracy ({nm})      & ${st.mean(a):.2f}\\% \\pm {POP(a):.2f}$~pp   "
          f"[BA {st.mean(ba):.2f}±{POP(ba):.2f}, F1 {st.mean(f1):.2f}±{POP(f1):.2f}, "
          f"recall {st.mean(rc):.2f}±{POP(rc):.2f}, spec {st.mean(sp):.2f}±{POP(sp):.2f}, "
          f"range {min(a):.2f}--{max(a):.2f}]")
