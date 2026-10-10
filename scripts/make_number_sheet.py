"""Generate the number sheet for the corrected GRSL manuscript.

Every value comes from the promoted artifacts (tag=grp after the 2026-10-10
fix re-run) — nothing here is typed by hand, so the sheet can be diffed against
the manuscript line by line.

Usage: python scripts/make_number_sheet.py [outfile]
"""
import json, csv, math, statistics as st, sys, datetime

ROOT = r"E:/Claude code/project/noise-label-cloud"
OUT = sys.argv[1] if len(sys.argv) > 1 else r"E:/Claude code/paper/noise_label_grsl/number_sheet_20261010.md"
WARM, STDMAX = 264.0, 1.5

T = json.load(open(rf"{ROOT}/output/paper_tables_grp.json"))
FZ = json.load(open(rf"{ROOT}/output/frozen_predictions/all_frozen_predictions_v2_grp.json"))
D = json.load(open(rf"{ROOT}/output/derived_tables_grp.json"))
paired = json.load(open(rf"{ROOT}/output/label_source_paired_5seed_grp.json"))
abl = json.load(open(rf"{ROOT}/output/2x2_ablation_grp.json"))
M = {r["fname"]: r for r in csv.DictReader(open(rf"{ROOT}/output/manifest/sample_manifest_grp.csv"))}

tr = {d["fname"]: d["truth"] for d in FZ["CLDMSK_raw"]}
test = sorted(tr)
cloud = [f for f in test if tr[f] == 1]
warm_c = [f for f in cloud if float(M[f]["m15_at_radar"]) >= WARM]
cold_c = [f for f in cloud if float(M[f]["m15_at_radar"]) < WARM]
warm_x = [f for f in test if tr[f] == 0 and float(M[f]["m15_at_radar"]) >= WARM]
cold_x = [f for f in test if tr[f] == 0 and float(M[f]["m15_at_radar"]) < WARM]

L = []
L.append(f"# GRSL number sheet — corrected re-run (generated {datetime.datetime.now():%Y-%m-%d %H:%M})")
L.append("")
L.append("Source: `output/paper_tables_grp.json`, `output/derived_tables_grp.json`, "
         "`output/frozen_predictions/all_frozen_predictions_v2_grp.json`, "
         "`output/label_source_paired_5seed_grp.json`, `output/2x2_ablation_grp.json` "
         "(all promoted from the 2026-10-10 `fix` re-run; the submitted versions are archived in "
         "`output/_archive_submitted_grp_20261010/`).")
L.append("")


def fmt(v, nd=1):
    return f"{v*100:.{nd}f}" if v <= 1.0000001 and nd else f"{v:.1f}"


def mcn(a, b, fs):
    x = sum(1 for f in fs if a[f][0] == a[f][1] and b[f][0] != b[f][1])
    y = sum(1 for f in fs if a[f][0] != a[f][1] and b[f][0] == b[f][1])
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


def strat(name):
    r = {d["fname"]: (d["pred"], d["truth"]) for d in FZ[name]}
    def c(fs, want):
        return sum(1 for f in fs if r[f][0] == want)
    return r, dict(warm_cloud_tp=c(warm_c, 1), cold_cloud_tp=c(cold_c, 1),
                   warm_clear_fp=c(warm_x, 1), cold_clear_fp=c(cold_x, 1))


L.append("## 1. Table I (n=197, radar-reference test set)")
L.append("")
L.append("| row | acc % [95% CI] | BA | F1 | MCC | warm-cloud TP /26 | cold-cloud TP /32 | FP warm-clear /119 | FP cold-clear /20 |")
L.append("|---|---|---|---|---|---|---|---|---|")
order = ["Always_clear", "Always_cloud", "CLDMSK_raw", "PureM15_T264", "PureM15_T266",
         "PhysPriorLabel_conservative", "PhysPriorLabel_moderate", "PhysPriorLabel_aggressive",
         "Standard_training", "LossCorrection", "CoTeaching_A", "CoTeaching_B", "GCE_q07", "Mixup",
         "PhysPrior_DL_3ch", "PhysPrior_CoTeaching_ensemble"]
for k in order:
    m = T["table1"][k]["metrics"]
    ci = m.get("ci", {})
    r, s = strat(k)
    L.append(f"| {T['table1'][k]['label'] if 'label' in T['table1'][k] else k} | "
             f"{m['acc']*100:.2f} [{ci.get('lo', float('nan'))*100:.1f}, {ci.get('hi', float('nan'))*100:.1f}] | "
             f"{m['balanced_accuracy']*100:.1f} | {m['f1']*100:.1f} | {m['mcc']:+.3f} | "
             f"{s['warm_cloud_tp']} | {s['cold_cloud_tp']} | {s['warm_clear_fp']} | {s['cold_clear_fp']} |")
L.append("")
L.append("Note: `±` for five-seed rows uses the population sd (ddof = 0) over the five seeds, "
         "the same convention as the submitted manuscript (70.66 ± 3.60 = 4.02 sample sd × √(4/5)).")
L.append("")

for key, lab in (("table_five_seed_physprior", "PhysPrior + DL, five seeds (25-epoch budget)"),
                 ("table_five_seed_coteaching", "PhysPrior + Co-Teaching, five seeds")):
    ps = T[key]["per_seed"]
    acc = [r["acc"] * 100 for r in ps]; ba = [r["balanced_accuracy"] * 100 for r in ps]
    f1 = [r["f1"] * 100 for r in ps]; rc = [r["recall"] * 100 for r in ps]; sp = [r["specificity"] * 100 for r in ps]
    pop = lambda v: st.stdev(v) * math.sqrt((len(v) - 1) / len(v))
    L.append(f"**{lab}**: acc {st.mean(acc):.2f} ± {pop(acc):.2f} (sample sd {st.stdev(acc):.2f}, "
             f"range {min(acc):.2f}–{max(acc):.2f}) | BA {st.mean(ba):.2f} ± {pop(ba):.2f} | "
             f"F1 {st.mean(f1):.2f} ± {pop(f1):.2f} | recall {st.mean(rc):.2f} ± {pop(rc):.2f} | "
             f"specificity {st.mean(sp):.2f} ± {pop(sp):.2f}")
    L.append("")

L.append("## 2. Paired-seed label-source block (fixed 20-epoch budget, no early stopping, seeds 0–4)")
L.append("")
R = {(r["label_mode"], r["n_channels"], r["seed"]): {d["fname"]: (d["pred"], d["truth"]) for d in r["test_records"]}
     for r in paired}
seeds = sorted({k[2] for k in R})
L.append("| label source | ch | acc mean ± sd | range | BA | cloud recall | warm-cloud TP /26 |")
L.append("|---|---|---|---|---|---|---|")
for mode in ("raw", "physprior", "m15thresh"):
    for ch in (2, 3):
        ks = [(mode, ch, s) for s in seeds]
        def col(fs, name):
            out = []
            for k in ks:
                rec = R[k]; tp = sum(1 for f in fs if rec[f] == (1, 1)); fp = sum(1 for f in fs if rec[f] == (1, 0))
                fn = sum(1 for f in fs if rec[f] == (0, 1)); tn = sum(1 for f in fs if rec[f] == (0, 0))
                out.append({"acc": (tp + tn) / len(fs) * 100, "recall": tp / max(tp + fn, 1) * 100,
                            "ba": (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2 * 100,
                            "tp": tp}[name])
            return out
        a = col(test, "acc"); ba = col(test, "ba"); rc = col(test, "recall"); w = col(warm_c, "tp")
        L.append(f"| {mode} | {ch} | {st.mean(a):.2f} ± {st.stdev(a):.2f} | {min(a):.2f}–{max(a):.2f} | "
                 f"{st.mean(ba):.2f} ± {st.stdev(ba):.2f} | {st.mean(rc):.2f} ± {st.stdev(rc):.2f} | "
                 f"{st.mean(w):.1f} ± {st.stdev(w):.1f} |")
L.append("")
L.append("Per-seed paired McNemar (same seed, same budget, only the label source differs):")
L.append("")
L.append("| comparison | ch | seed-by-seed (win-only counts, exact p) | seeds won |")
L.append("|---|---|---|---|")
for a, b, lab in (("physprior", "raw", "PhysPrior labels vs raw CLDMSK"),
                  ("m15thresh", "physprior", "M15-threshold labels vs PhysPrior labels"),
                  ("m15thresh", "raw", "M15-threshold labels vs raw CLDMSK")):
    for ch in (2, 3):
        cells = []
        wins = 0
        for s in seeds:
            x, y, p = mcn(R[(a, ch, s)], R[(b, ch, s)], test)
            cells.append(f"{x}-{y} p={p:.4f}")
            wins += int(x > y)
        L.append(f"| {lab} (overall 197) | {ch} | {'; '.join(cells)} | {wins}/5 |")
L.append("")
L.append("Restricted to the 26 warm-top cloud pixels (M15 ≥ 264 K at the radar pixel):")
L.append("")
for a, b, lab in (("physprior", "m15thresh", "PhysPrior vs M15-threshold labels"),
                  ("physprior", "raw", "PhysPrior vs raw labels")):
    for ch in (2, 3):
        cells = []; sig = 0
        for s in seeds:
            x, y, p = mcn(R[(a, ch, s)], R[(b, ch, s)], warm_c)
            cells.append(f"{x}-{y} p={p:.4f}")
            sig += int(p < 0.05)
        L.append(f"- {lab}, {ch} ch: {'; '.join(cells)} — significant in {sig}/5 seeds")
L.append("")

L.append("## 3. 2×2 channel–label cells under the corrected path (seed 42, 20 epochs)")
L.append("")
for c in abl:
    m = c["test"]
    L.append(f"- {c['label_mode']:10s} {c['n_channels']} ch: acc {m['acc']*100:.2f}%  BA {m['balanced_accuracy']*100:.2f}  "
             f"recall {m['recall']*100:.1f}  spec {m['specificity']*100:.1f}  MCC {m['mcc']:+.3f}")
d = T["table_ablation_deltas"]
L.append(f"- deltas (single seed): labels {d['physprior_effect_2ch_pp']:+.2f} / {d['physprior_effect_3ch_pp']:+.2f} pp, "
         f"basemap {d['basemap_effect_raw_pp']:+.2f} / {d['basemap_effect_physprior_pp']:+.2f} pp")


def acc_of(rec, fs):
    return sum(1 for f in fs if rec[f][0] == rec[f][1]) / len(fs) * 100


mean_acc = lambda mode, ch: st.mean([acc_of(R[(mode, ch, s)], test) for s in seeds])
L.append(f"- five-seed mean effects: labels {mean_acc('physprior', 2) - mean_acc('raw', 2):+.2f} pp (2 ch) / "
         f"{mean_acc('physprior', 3) - mean_acc('raw', 3):+.2f} pp (3 ch); "
         f"basemap {mean_acc('raw', 3) - mean_acc('raw', 2):+.2f} pp (raw) / "
         f"{mean_acc('physprior', 3) - mean_acc('physprior', 2):+.2f} pp (PhysPrior)")
for ch in (2, 3):
    ds_ = [acc_of(R[("physprior", ch, s)], test) - acc_of(R[("raw", ch, s)], test) for s in seeds]
    L.append(f"  - paired label effect, {ch} ch, per seed: " + ", ".join(f"{v:+.2f}" for v in ds_) +
             f" → mean {st.mean(ds_):+.2f} ± {st.stdev(ds_):.2f} pp, "
             f"{sum(1 for v in ds_ if v > 0)}/5 seeds positive")
L.append("")

L.append("## 4. Direct-rule confusion, flips and strata (unchanged by the fixes: no training)")
L.append("")
L.append(f"- direct PhysPrior moderate confusion (TN/FP/FN/TP): "
         f"{D['direct_rule_confusion']['tn']}/{D['direct_rule_confusion']['fp']}/{D['direct_rule_confusion']['fn']}/{D['direct_rule_confusion']['tp']}")
r_pp, s_pp = strat("PhysPriorLabel_moderate"); r_m15, s_m15 = strat("PureM15_T264")
x, y, p = mcn(r_pp, r_m15, cloud)
xw, yw, pw = mcn(r_pp, r_m15, warm_c)
L.append(f"- label-level paired McNemar vs the matched M15 threshold: cloudy pixels {x}-{y} p={p:.5f}; "
         f"warm-top clouds {xw}-{yw} p={pw:.5f}")
r_raw, s_raw = strat("CLDMSK_raw")
L.append(f"- raw apparent FP 107 decomposes as {s_raw['warm_clear_fp']} warm "
         f"({s_raw['warm_clear_fp']}/{len(warm_x)}) + {s_raw['cold_clear_fp']} radiatively cooled "
         f"({s_raw['cold_clear_fp']}/{len(cold_x)}); the cooled {len(cold_x)} are mislabelled by every "
         f"M15-based rule (PhysPrior {s_pp['cold_clear_fp']}/{len(cold_x)}, M15 threshold {s_m15['cold_clear_fp']}/{len(cold_x)})")
L.append("")
L.append("## 5. Station / season / agreement diagnostics (Table III, IV, VI)")
L.append("")
L.append("```json")
L.append(json.dumps(D, indent=1)[:2600])
L.append("```")
L.append("")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT, "|", len(L), "lines")
