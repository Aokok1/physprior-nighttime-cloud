"""Check every number the manuscript reports against the artifacts that produce it.

The gate is deliberately one-directional: it derives an expected value from a
JSON artifact and asserts the manuscript contains that value, formatted the way
the manuscript formats it. It also asserts that no *stale* value survives, which
catches a partial update after a re-run.

Expected strings track the manuscript, not the artifacts' internal layout. As of
the 2026-10-10 restructure:

  * the per-station and per-season floats and the agreement/five-seed float are
    gone; those numbers are now stated in prose (the intervals and the combined
    precision/recall/specificity are deferred to supplementary S1.8 and are
    therefore *not* assertable here -- see the "not asserted" block below),
  * the 2x2 single-seed ablation float is now a 3-label-source x 2-channel table
    of five paired seeds (output/label_source_paired_5seed_grp.json), whose
    "warm" column counts the 26 radar-cloudy test pixels with M15 >= 264 K
    (manifest m15_at_radar, truth from the v2 frozen prediction records),
  * seed spreads are printed as the *population* sd (ddof = 0), the convention
    rebuild_paper_tables.py writes into paper_tables_*_grp.json and the
    convention the ablation table's caption states (sd_disp below).

Run after `rebuild_paper_tables.py` and `derive_station_season_tables.py`:

    python -X utf8 scripts/verify_manuscript_numbers.py

Exit code 0 = every checked number is present and no stale number survives.
"""
import io
import json
import math
import os
import re
import statistics as st
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"E:/Claude code/project/noise-label-cloud"
TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"
TAG = os.environ.get("NL_CKPT_TAG", "grp")
SUF = f"_{TAG}" if TAG else ""
# point the gate at an archived artifact set to audit a previous manuscript state
ARTIFACT_DIR = os.environ.get("NL_ARTIFACT_DIR", os.path.join(ROOT, "output"))


def load(rel):
    p = os.path.join(ARTIFACT_DIR, rel)
    if not os.path.exists(p):
        raise SystemExit(f"missing artifact: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


t1 = load(f"paper_tables{SUF}.json")["table1"]
five = load(f"paper_tables{SUF}.json")["table_five_seed_physprior"]
ct5 = load(f"paper_tables{SUF}.json")["table_five_seed_coteaching"]
dv = load(f"derived_tables{SUF}.json")
paired = load(f"label_source_paired_5seed{SUF}.json")
x2 = load(f"2x2_ablation{SUF}.json")
frozen = load(f"frozen_predictions/all_frozen_predictions_v2{SUF}.json")

# The pre-fix schema exported `each_correct` (the MODEL's count) and swapped the
# two `*_only_correct` keys. Normalise it so this gate can audit the manuscript
# state that was written against it.
_ag = dv.get("agreement", {})
if "rule_correct" not in _ag:
    _ag["model_correct"] = _ag["each_correct"]
    _ag["rule_only_correct"], _ag["dl_only_correct"] = (
        _ag["dl_only_correct"], _ag["rule_only_correct"])
    _ag["rule_correct"] = _ag["both_correct"] + _ag["rule_only_correct"]
    print("note: derived table uses the pre-fix agreement schema; normalised", file=sys.stderr)

tex = open(TEX, encoding="utf-8").read()
# Collapse horizontal whitespace so a check can quote a whole table row without
# reproducing the source's alignment padding. Every expected string is compared
# against this normalised text, so a check may span a label and its value.
tex_norm = re.sub(r"[ \t]+", " ", tex)


def acc(key):
    return t1[key]["metrics"]["acc"] * 100


def pct(x, nd=1):
    return f"{x:.{nd}f}"


def ci(key):
    c = t1[key]["metrics"]["ci"]
    return f"{c['lo'] * 100:.1f}", f"{c['hi'] * 100:.1f}"


def pop_sd(v):
    """Population sd (ddof = 0), the convention the manuscript prints."""
    return st.stdev(v) * math.sqrt((len(v) - 1) / len(v)) if len(v) > 1 else 0.0


def sd_disp(v, nd=1):
    """Format a seed spread the way the manuscript prints it: the population sd
    (ddof = 0), rounded once. The table's own caption states the convention."""
    return f"{pop_sd(v):.{nd}f}"


CHECKS = []          # (label, expected substring)
STALE = []           # (label, substring that must NOT appear)
NOT_IN_TEX = []      # (label, value) facts the revision removed from the manuscript
INNER = []           # (label, bool) artifact-internal consistency, no tex involved


def want(label, s):
    CHECKS.append((label, s))


def stale(label, s):
    STALE.append((label, s))


def unasserted(label, s):
    NOT_IN_TEX.append((label, s))


def inner(label, ok):
    INNER.append((label, bool(ok)))


# ---- Table I rows ----------------------------------------------------------
for key, nd in (("CLDMSK_raw", 1), ("Standard_training", 1), ("LossCorrection", 1),
                ("CoTeaching_A", 1), ("CoTeaching_B", 1), ("GCE_q07", 1),
                ("Mixup", 1), ("PhysPrior_CoTeaching_ensemble", 1),
                ("PureM15_T264", 1), ("PureM15_T266", 1),
                ("PhysPriorLabel_conservative", 1), ("PhysPriorLabel_moderate", 1),
                ("PhysPriorLabel_aggressive", 1), ("PhysPrior_DL_3ch", 1)):
    v = pct(acc(key), nd)
    want(f"Table I {key} accuracy", v + "\\%")

lo, hi = ci("PhysPrior_DL_3ch")
want("Table I PhysPrior+DL CI", f"[{lo},\\,{hi}]")

# ---- Table III: label source x channels, five paired seeds ------------------
# Rebuild every cell from the 30 run records the same way the table was
# generated (scripts/emit_tex_rows.py): recompute accuracy and the warm-top
# recovery count per seed, then take the seed mean and the population sd.
import csv  # noqa: E402

MAN_PATH = os.path.join(ROOT, "output", "manifest",
                        "sample_manifest_grp.csv" if TAG == "grp" else "sample_manifest.csv")
man = list(csv.DictReader(open(MAN_PATH, encoding="utf-8", newline="")))
man_by_fname = {r["fname"]: r for r in man}

RUN = {(r["label_mode"], r["n_channels"], r["seed"]): r for r in paired}
SEEDS = sorted({r["seed"] for r in paired})
MODES = ("raw", "physprior", "m15thresh")
CHANS = (2, 3)
# the "warm" stratum: radar-cloudy test pixels whose M15 at the radar pixel is at
# or above the moderate preset's own boundary, i.e. clouds a 264 K threshold cannot label
WARM_T = 264.0
warm_fnames = {d["fname"] for d in frozen["CLDMSK_raw"]
               if d["truth"] == 1 and float(man_by_fname[d["fname"]]["m15_at_radar"]) >= WARM_T}

PACC, PWARM = {}, {}
for mode in MODES:
    for ch in CHANS:
        recs = [RUN[(mode, ch, s)] for s in SEEDS]
        PACC[(mode, ch)] = [r["test"]["acc"] for r in recs]
        PWARM[(mode, ch)] = [sum(1 for d in r["test_records"]
                                 if d["pred"] == 1 and d["fname"] in warm_fnames) for r in recs]
# the table bolds the best accuracy per channel set; derive it, do not assume it
BEST = {ch: max(MODES, key=lambda m: st.mean(PACC[(m, ch)])) for ch in CHANS}
ROWNAME = {"raw": "Raw CLDMSK", "physprior": "PhysPrior moderate", "m15thresh": "M15 threshold mask"}

for mode in MODES:
    cells = []
    for ch in CHANS:
        a = st.mean(PACC[(mode, ch)])
        shown = f"\\textbf{{{a:.1f}}}" if BEST[ch] == mode else f"{a:.1f}"
        w = st.mean(PWARM[(mode, ch)])
        cells.append(f"{shown} $\\pm$ {sd_disp(PACC[(mode, ch)])} & {w:.1f} $\\pm$ {sd_disp(PWARM[(mode, ch)])}")
    want(f"Table III {mode} row (acc+warm, 2ch/3ch)", f"{ROWNAME[mode]} & " + " & ".join(cells))


def paired_delta(a, b, ch=None):
    """Per-seed differences of two cells, as the effect rows print them."""
    if ch is not None:
        return [PACC[(a, ch)][i] - PACC[(b, ch)][i] for i in range(len(SEEDS))]
    return [PACC[(a, 3)][i] - PACC[(a, 2)][i] for i in range(len(SEEDS))]


def effect_pair(d_first, d_second):
    """The two \\multicolumn cells an effect row prints, left (2-ch) then right (3-ch)."""
    out = []
    for d in (d_first, d_second):
        out.append("\\multicolumn{2}{r}" + "{$%s \\pm %s$}" % (f"{st.mean(d):+.1f}", sd_disp(d)))
    return " & ".join(out)


lab2, lab3 = paired_delta("physprior", "raw", 2), paired_delta("physprior", "raw", 3)
bm_raw, bm_pp = paired_delta("raw", "raw"), paired_delta("physprior", "physprior")
thr2, thr3 = paired_delta("m15thresh", "physprior", 2), paired_delta("m15thresh", "physprior", 3)

want("Table III label effect row",
     "label effect, PhysPrior $-$ raw & " + effect_pair(lab2, lab3))
want("Table III basemap effect row",
     "basemap effect, raw / PhysPrior labels & " + effect_pair(bm_raw, bm_pp))

# the same paired effects are restated in the abstract, the contribution list,
# the ablation section and the conclusion; check each restatement, since a
# re-run that updates only the table is exactly what this gate exists to catch.
LAB = f"${st.mean(lab2):+.1f} \\pm {sd_disp(lab2)}$"
LAB3 = f"${st.mean(lab3):+.1f} \\pm {sd_disp(lab3)}$"
want("abstract label gain",
     f"the label factor is worth {LAB} percentage points (pp) at two channels and {LAB3}~pp at three")
want("intro contribution label gain",
     f"Replacing raw labels with corrected ones is worth {LAB}~pp at two input channels and {LAB3}~pp at three")
want("ablation-section label gain and seed count",
     f"the label factor is large and seed-stable: {LAB}~pp at two channels and {LAB3}~pp at three, "
     f"positive in ${sum(1 for v in lab2 if v > 0)}/{len(SEEDS)}$ seeds")
want("five-seed protocol stated in the table's section",
     f"train a fixed ${RUN[('raw', 2, 0)]['epochs']}$ epochs with no early stopping, "
     f"and repeat seeds $0$--${SEEDS[-1]}$")

m15a2 = st.mean(PACC[("m15thresh", 2)])
wraw2, wraw3 = st.mean(PWARM[("raw", 2)]), st.mean(PWARM[("raw", 3)])
wpp2, wpp3 = st.mean(PWARM[("physprior", 2)]), st.mean(PWARM[("physprior", 3)])
wm2, wm3 = st.mean(PWARM[("m15thresh", 2)]), st.mean(PWARM[("m15thresh", 3)])
want("abstract threshold-label cell",
     f"trains a more accurate network (${m15a2:.2f}\\% \\pm {sd_disp(PACC[('m15thresh', 2)], 2)}$~pp) "
     f"that recovers only ${wm2:.1f} \\pm {sd_disp(PWARM[('m15thresh', 2)])}$ of the ${len(warm_fnames)}$ "
     f"clouds warmer than ${WARM_T:.0f}$~K against "
     f"${wpp2:.1f} \\pm {sd_disp(PWARM[('physprior', 2)])}$")
want("ablation threshold-vs-corrected accuracy",
     f"the threshold mask is the better \\emph{{accuracy}} trainer "
     f"($+{st.mean(thr2):.1f}$ and $+{st.mean(thr3):.1f}$~pp over corrected labels, also ${len(SEEDS)}/{len(SEEDS)}$)")
want("ablation threshold-vs-corrected warm recovery",
     f"recovers ${wm2:.1f} \\pm {sd_disp(PWARM[('m15thresh', 2)])}$ of the ${len(warm_fnames)}$ warm-top clouds "
     f"against ${wpp2:.1f} \\pm {sd_disp(PWARM[('physprior', 2)])}$ at two channels and "
     f"${wm3:.1f} \\pm {sd_disp(PWARM[('m15thresh', 3)])}$ against "
     f"${wpp3:.1f} \\pm {sd_disp(PWARM[('physprior', 3)])}$ at three")
want("ablation basemap trade-off on warm recovery",
     f"(${wraw2:.1f} \\to {wraw3:.1f}$ raw, ${wpp2:.1f} \\to {wpp3:.1f}$ corrected)")
want("discussion threshold blind spot",
     f"its network recovers ${wm2:.1f}$--${wm3:.1f}$ of them against "
     f"${wpp2:.1f}$--${wpp3:.1f}$ for corrected labels")
want("discussion threshold-label advantage",
     f"the same threshold trains a network ${st.mean(thr2):.1f}$--${st.mean(thr3):.1f}$~pp more accurate still")
inner(f"Table III protocol: {len(paired)} runs, {len(SEEDS)} seeds, one epoch budget, n=197",
      len(paired) == len(MODES) * len(CHANS) * len(SEEDS)
      and {r["epochs"] for r in paired} == {20}
      and {r["test"]["n"] for r in paired} == {197})


# ---- five seeds ------------------------------------------------------------
# The agreement/five-seed float was removed as a table; section III-G now states
# each statistic in prose as "<metric> $mean \pm sd$" (no \% except on accuracy).
want("five-seed PhysPrior mean", f"{five['accuracy_mean'] * 100:.2f}\\% \\pm {five['accuracy_sd'] * 100:.2f}")
want("five-seed Co-Teaching mean", f"{ct5['accuracy_mean'] * 100:.2f}\\% \\pm {ct5['accuracy_sd'] * 100:.2f}")
for m, texname in (("balanced_accuracy", "balanced accuracy"), ("f1", "F1"),
                   ("recall", "recall"), ("specificity", "specificity")):
    want(f"five-seed {texname}",
         f"{texname} ${five[m + '_mean'] * 100:.2f} \\pm {five[m + '_sd'] * 100:.2f}$")
accs = [p["acc"] * 100 for p in five["per_seed"]]
want("five-seed range", f"${min(accs):.2f}$--${max(accs):.2f}$")
want("five-seed gap over the raw-label row",
     f"${five['accuracy_mean'] * 100 - acc('CLDMSK_raw'):.1f}$~pp above the raw-CLDMSK label row")
want("five-seed mean recall in the always-clear comparison",
     f"mean recall of ${five['recall_mean'] * 100:.1f}\\%$")
want("conclusion Co-Teaching increment",
     f"Co-Teaching adds ${ct5['accuracy_mean'] * 100 - five['accuracy_mean'] * 100:.2f}$~pp")

# ---- derived tables --------------------------------------------------------
# The agreement float is gone; section III-G carries the same facts in one
# sentence. The labels stay inside the expected strings: the pre-fix derived
# table swapped rule_only/dl_only, and a value-only check cannot tell a correct
# triple from a swapped one.
ag = dv["agreement"]
N_AG = ag["n"]
want("agreement share + labelled counts (prose)",
     f"on ${ag['agree'] / N_AG * 100:.2f}\\%$ of the ${N_AG}$ pixels "
     f"($\\kappa = +{ag['kappa']:.3f}$; ${ag['both_correct']}$ both correct, "
     f"${ag['rule_only_correct']}$ rule only, ${ag['dl_only_correct']}$ DL only)")
want("mcnemar discordant pair and p (prose)",
     f"clear in ${ag['mcnemar_b']}$ pixels and the reverse in only ${ag['mcnemar_c']}$ "
     f"(exact McNemar $p = {ag['mcnemar_p']:.3f}$)")
# rule_correct / model_correct / both_wrong no longer appear in the manuscript, but
# the prose triple pins them down: keep them as artifact-internal consistency.
inner("agreement block closes (rule = both + rule-only)",
      ag["rule_correct"] == ag["both_correct"] + ag["rule_only_correct"])
inner("agreement block closes (model = both + DL-only)",
      ag["model_correct"] == ag["both_correct"] + ag["dl_only_correct"])
inner("agreement block closes (both_wrong = n - rule - DL-only)",
      ag["both_wrong"] == N_AG - ag["rule_correct"] - ag["dl_only_correct"])
inner("agreement block closes (agree = both + both_wrong)",
      ag["agree"] == ag["both_correct"] + ag["both_wrong"])
inner("DL-correct count equals the Table I PhysPrior+DL row",
      round(ag["model_correct"] / N_AG * 100, 1) == round(acc("PhysPrior_DL_3ch"), 1))
for _lab, _v in (("rule correct", ag["rule_correct"]), ("model correct", ag["model_correct"]),
                 ("both incorrect", ag["both_wrong"])):
    unasserted(f"agreement {_lab} (was Table VI row)",
               f"{_v}/{N_AG} ({_v / N_AG * 100:.2f}\\%)  -- no longer stated in the manuscript")

dc = dv["direct_rule_confusion"]
want("direct-rule confusion counts",
     f"${dc['tn']}/{dc['fp']}/{dc['fn']}/{dc['tp']}$")

# ---- per station and per season --------------------------------------------
# Both floats were removed; section III-F now states the checkpoint's split in
# prose (n, accuracy, recall, specificity per station; accuracy and n per
# season). The bootstrapped per-season intervals and the combined
# precision/recall/specificity were delegated to supplementary S1.8.
station = dv["per_station"]
cs, lm, comb = station["CS"], station["LM"], station["combined"]
want("per-station Changsha (prose)",
     f"Changsha (urban) $n{{=}}{cs['n']}$, accuracy ${cs['acc'] * 100:.1f}\\%$, "
     f"recall ${cs['recall'] * 100:.1f}\\%$, specificity ${cs['specificity'] * 100:.1f}\\%$")
want("per-station Longmen (prose)",
     f"Longmen (suburban) $n{{=}}{lm['n']}$, ${lm['acc'] * 100:.1f}\\%$, "
     f"recall ${lm['recall'] * 100:.1f}\\%$, specificity ${lm['specificity'] * 100:.1f}\\%$")
want("per-station combined (prose)",
     f"combined $n{{=}}{comb['n']}$, ${comb['acc'] * 100:.1f}\\%$")
want("per-station Changsha subset in the agreement section",
     f"(${cs['recall'] * 100:.0f}\\%$ recall, ${cs['specificity'] * 100:.1f}\\%$ specificity)")
want("degenerate split caveat ties station to season",
     f"all ${cs['n']}$ Changsha samples are winter and all ${lm['n']}$ Longmen samples "
     f"are spring, summer or autumn")
inner("Changsha n equals the winter n (degenerate split claim)",
      cs["n"] == dv["per_season"]["Winter"]["n"])
inner("station counts sum to the combined row",
      cs["n"] == comb["n"] - lm["n"] and round(comb["acc"] * 100, 1) == round(acc("PhysPrior_DL_3ch"), 1))
for _lab, _v in (("precision", comb["precision"]), ("recall", comb["recall"]),
                 ("specificity", comb["specificity"])):
    unasserted(f"combined {_lab} (was Table IV row)",
               f"{_v * 100:.1f}\\%  -- no longer stated in the manuscript (deferred to S1.8)")

MONTHS = {"Spring": "Mar--May", "Summer": "Jun--Aug", "Autumn": "Sep--Nov", "Winter": "Dec--Feb"}
for season in ("Spring", "Summer", "Autumn", "Winter"):
    m = dv["per_season"][season]
    want(f"per-season {season} (prose)",
         f"{season.lower()} ({MONTHS[season]}, $n{{=}}{m['n']}$) ${m['acc'] * 100:.1f}\\%$")
    unasserted(f"per-season {season} 95% interval",
               f"[{m['ci']['lo'] * 100:.1f}, {m['ci']['hi'] * 100:.1f}] "
               f"-- no longer stated in the manuscript (deferred to S1.8)")
# the within-Longmen contrast is quoted twice (section III-F and the discussion)
want("within-Longmen seasonal contrast",
     f"(${dv['per_season']['Summer']['acc'] * 100:.1f}\\%$ summer against "
     f"${dv['per_season']['Autumn']['acc'] * 100:.1f}\\%$ autumn)")


# ---- development set (160 pixels, deployed operator) ------------------------
# Verified two independent ways before being written down: from the manifest
# columns m15_at_radar / m15_local_std, and by re-reading the 160 .npz files and
# recomputing M15 and the 5x5 population sigma from scratch. The two paths agree
# to 0.0000 K, giving raw CLDMSK 101/160 = 63.12%, moderate 55.62%, and a grid
# optimum of 63.12% at (270, 0.5) that merely ties raw CLDMSK.
want("development-set preset accuracy", r"neither it ($55.6\%$)")
want("development-set grid best and raw CLDMSK",
     r"threshold grid ($63.1\%$) improves on raw CLDMSK ($63.1\%$)")

# ---- prevalence ------------------------------------------------------------
truths = [r["truth"] for r in frozen["CLDMSK_raw"]]
n_clear = sum(1 for t in truths if t == 0)
want("always-clear accuracy", f"{n_clear / len(truths) * 100:.1f}\\%")

# ---- prevalence and split closure (from the manifest) ----------------------
# `man` / MAN_PATH were read with the ablation block above: the manifest is an
# input, not a re-run artifact, so it is always read from the live output
# directory even when the gate audits an archived artifact set.
by_split = {}
for r in man:
    by_split.setdefault(r["split"], []).append(r)
n_test = len(by_split["test"])
n_clear = sum(1 for r in by_split["test"] if float(r["radar_label"]) == 0)
n_cloud = n_test - n_clear
n_train = len(by_split["train_grp"])
n_val = len(by_split["val_grp"])
inner("frozen records and manifest agree on the test set",
      len(truths) == n_test and n_clear == sum(1 for t in truths if t == 0))
inner("warm-top stratum is the cloudy subset with M15 >= 264 K",
      0 < len(warm_fnames) < n_cloud)


def texnum(n):
    """1234 -> '1{,}234' as the manuscript writes it."""
    return f"{n:,}".replace(",", "{,}")


want("test prevalence",
     f"${n_clear}$ (${n_clear / n_test * 100:.1f}$\\%) are radar-clear and "
     f"${n_cloud}$ (${n_cloud / n_test * 100:.1f}$\\%) are radar-cloudy")
want("overpass counts",
     f"${texnum(len({r['overpass_group'] for r in by_split['train_grp']}))}$ training, "
     f"${texnum(len({r['overpass_group'] for r in by_split['val_grp']}))}$ validation and "
     f"${texnum(len({r['overpass_group'] for r in by_split['test']}))}$ test overpasses")
want("total patches", f"the ${texnum(len(man))}$ samples are two time series")
want("split sizes",
     f"${texnum(n_train)}$ train $+$ ${texnum(n_val)}$ validation $+$ "
     f"${texnum(n_test)}$ radar-reference test")

# ---- values from the pre-determinism run that must not survive -------------
# Each fragment carries enough LaTeX context to be unambiguous: a bare "43.1" is
# also a legitimate interval endpoint in the new tables, so the value alone
# cannot be forbidden. Presence of any of these means an edit was applied in one
# place and missed in another.
for _label, _s in [
    ("five-seed PhysPrior mean (old)", r"$70.66\% \pm 3.74$"),
    ("five-seed Co-Teaching mean (old)", r"$70.96\% \pm 2.64$"),
    ("five-seed SD (old)", r"$3.74$~pp"),
    ("five-seed BA (old)", r"$73.88\% \pm 1.68$"),
    ("five-seed F1 (old)", r"$62.25\% \pm 2.03$"),
    ("five-seed recall (old)", r"$81.72\% \pm 5.73$"),
    ("five-seed specificity (old)", r"$66.04\% \pm 7.33$"),
    ("five-seed range (old)", r"$66.50$--$77.16$"),
    ("ablation delta 2ch (old)", r"$21.3$~pp"),
    ("ablation delta 3ch (old)", r"$31.5$~pp"),
    ("ablation delta 2ch (det, old)", r"$12.7$~pp"),
    ("ablation delta 3ch (det, old)", r"$36.5$~pp"),
    ("ablation 2ch/3ch phrasing (old)", r"with two input channels and $36.5$~pp with three"),
    ("basemap effect raw (old)", r"$-3.55$~pp"),
    ("basemap effect PhysPrior (old)", r"$+6.60$~pp"),
    ("PhysPrior effect 2ch (old)", r"$+21.32$~pp"),
    ("PhysPrior effect 3ch (old)", r"$+31.47$~pp"),
    ("baseline range (old)", r"$32.5$--$47.2\%$"),
    ("agreement share (old)", r"$84.26\%$"),
    ("kappa (old)", r"$+0.684$"),
    ("McNemar counts (old)", r"$(19, 12)$"),
    ("McNemar p (old)", r"$0.281$"),
    ("raw-baseline gap (old)", r"$26.2$~pp"),
    ("winter accuracy (old)", r"$48.3\%"),
    ("autumn accuracy (old)", r"$85.0\%"),
    ("discussion lift chain (old)", r"from $43.1\%$ to $75.6\%$"),
    ("Table I loss correction row (old)", r"$36.0\%\,[29.4,\,42.6]"),
    ("Table I co-teaching A row (old)", r"$44.7\%\,[37.6,\,51.8]"),
    ("Table I GCE row (old)", r"$47.2\%\,[40.1,\,54.3]"),
    ("Table I mixup row (old)", r"$37.6\%\,[31.0,\,44.2]"),
    ("Table I PhysPrior ensemble row (old)", r"$66.0\%\,[59.4,\,72.6]"),
    ("Table I PhysPrior+DL row (old)", r"\textbf{75.6\%}\,[69.5,\,81.7]"),
    ("Table V raw cells (old)", r"47.7\%  & 44.2\%"),
    ("Table V PhysPrior cells (old, best-val)", r"69.0\%  & \textbf{75.6\%}"),
    ("Table V raw cells (old, det best-val)", r"49.7\%  & 35.0\%"),
    ("Table V PhysPrior cells (old, det best-val)", r"62.4\%  & \textbf{71.6\%}"),
    ("Table V basemap raw (old)", r"$-14.72$~pp"),
    ("Table V basemap PhysPrior (old)", r"$+9.14$~pp"),
    ("Table V label effect 2ch (old)", r"$+12.69$~pp"),
    ("Table V label effect 3ch (old)", r"$+36.55$~pp"),
    ("Table IV Changsha row (old)", r"& 29  & 48.3\% & 46.4\% & 100.0\% & 6.2\% \\"),
    ("Table IV Longmen row (old)", r"& 168 & 76.2\% & 53.7\% & 80.0\% & 74.8\% \\"),
    ("Table IV combined row (old)", r"& 197 & 72.1\% & 51.6\% & 84.5\% & 66.9\% \\"),
    ("Table IV-b spring (phase4)", r"& 72.2\% & [60.7, 82.6] & 56 \\"),
    ("Table IV-b summer (old)", r"& 73.6\% & [62.5, 83.3] & 72 \\"),
    ("Table IV-b autumn (old)", r"& 85.0\% & [72.5, 95.0] & 40 \\"),
    ("Table IV-b winter (old)", r"& 48.3\% & [31.0, 65.5] & 29 \\"),
    ("Table VI DL correct (old)", r"142/197 (72.08\%)"),
    ("Table VI both correct (old)", r"124/197 (62.94\%)"),
    ("Table VI rule only (old)", r"& 13/197 \\"),
    ("Table VI DL only (old)", r"& 18/197 \\"),
    ("Table VI both incorrect (old)", r"& 42/197 \\"),
    # development-set values computed over the pre-rebuild 167-patch set
    # (62.87% = 105/167 and 62.28% = 104/167 are exact fractions of 167, not of
    # the current 160, which is how the mismatch was caught)
    ("development preset (old 167-set)", r"neither it ($55.7\%$)"),
    ("development grid best (old 167-set)", r"(best $62.3\%$)"),
    ("development raw CLDMSK (old 167-set)", r"raw CLDMSK ($62.9\%$)"),
    ("Table VI agreement (old)", r"166/197 (84.26\%)"),
    ("Changsha cloud-biased claim (old)", r"(100\% recall, $6.2\%$ specificity)"),
    ("run-to-run spread claim (old)", r"moved individual seeds by up to $4$~pp"),
    ("co-teaching divergence claim (old)", r"the two Co-Teaching variants diverge sharply"),
]:
    stale(_label, _s)

# ---- values superseded by the 2026-10-10 re-run and restructure ------------
# The trained rows all moved when the two training-side data-path defects were
# fixed, and three floats became prose, so every value the *submitted* revision
# printed is forbidden here too. Same anchoring rule as above: the fragment must
# carry enough LaTeX context, because 48.2 vs 48.5 and 67.5 vs 67.51 are both
# real numbers in the current manuscript and a bare value would be a false alarm.
X2 = {(r["label_mode"], r["n_channels"]): r["test"]["acc"] * 100 for r in x2}
for _label, _s in [
    # Table I trained rows at the values they held before the fix (the label is
    # part of the pattern: Loss Correction now legitimately prints 48.7, which was
    # Co-Teaching A's old value).
    ("Table I standard-training row (pre-fix)", r"Standard training & 197 & 46.7\%"),
    ("Table I loss-correction row (pre-fix)", r"Loss Correction & 197 & 58.4\%"),
    ("Table I co-teaching A row (pre-fix)", r"Co-Teaching A & 197 & 48.7\%"),
    ("Table I co-teaching B row (pre-fix)", r"Co-Teaching B & 197 & 51.8\%"),
    ("Table I GCE row (pre-fix)", r"GCE ($q{=}0.7$) & 197 & 44.7\%"),
    ("Table I ensemble row bolded value (pre-fix)", r"\textbf{76.6\%}"),
    ("Table I PhysPrior+DL metrics (pre-fix)", r"& 73.0 & 61.1 & +0.420 \\"),
    ("Table I ensemble metrics (pre-fix)", r"& \textbf{75.4} & \textbf{64.6} & \textbf{+0.481}"),
    ("raw-label block range (pre-fix)", r"$29.4$--$58.4\%$"),
    ("raw-label prose gap (pre-fix)", r"$46.7\%$ on raw"),
    # five-seed block, printed as a table before the restructure
    ("five-seed mean (pre-fix)", r"$70.66\% \pm 3.60$"),
    ("five-seed Co-Teaching mean (pre-fix)", r"$73.81\% \pm 1.38$"),
    ("five-seed BA (pre-fix)", r"$73.88\% \pm 2.54$"),
    ("five-seed BA prose form (pre-fix)", r"$73.88 \pm 2.54$"),
    ("five-seed F1 (pre-fix)", r"$62.22\% \pm 2.84$"),
    ("five-seed F1 prose form (pre-fix)", r"$62.22 \pm 2.84$"),
    ("five-seed recall (pre-fix)", r"$81.72\% \pm 4.31$"),
    ("five-seed recall prose form (pre-fix)", r"$81.72 \pm 4.31$"),
    ("five-seed specificity (pre-fix)", r"$66.04\% \pm 5.82$"),
    ("five-seed specificity prose form (pre-fix)", r"$66.04 \pm 5.82$"),
    ("five-seed range (pre-fix)", r"$65.48$--$75.63$"),
    ("five-seed mean recall claim (pre-fix)", r"mean recall of $81.7\%$"),
    ("Co-Teaching increment (pre-fix)", r"Co-Teaching adds $3.15$~pp"),
    ("raw-baseline gap (pre-fix)", r"$26.5$~pp above the raw-CLDMSK"),
    # the single-seed 2x2 ablation float, at its pre-fix values ...
    ("2x2 raw cells (pre-fix)", r"Raw CLDMSK & 52.8\% & 48.2\%"),
    ("2x2 PhysPrior cells (pre-fix)", r"\textbf{72.6\%} & 67.5\%"),
    ("2x2 basemap raw effect (pre-fix)", r"$-4.57$~pp"),
    ("2x2 basemap PhysPrior effect (pre-fix)", r"$-5.08$~pp"),
    ("2x2 label effect 2ch (pre-fix)", r"$+19.80$~pp"),
    ("2x2 label effect 3ch (pre-fix)", r"$+19.29$~pp"),
    ("2x2 label effect prose (pre-fix)", r"$+19.8$~pp"),
    ("2x2 label effect prose 3ch (pre-fix)", r"$+19.3$~pp"),
    ("2x2 basemap cost prose (pre-fix)", r"costs $4.6$--$5.1$~pp"),
    ("2x2 abstract effect pair (pre-fix)", r"by $19.8$ and $19.3$~pp"),
    # ... and at the re-run values, which belong to a float the manuscript no longer has
    ("2x2 raw cells (re-run, float removed)", f"& {X2[('raw', 2)]:.1f}\\% & {X2[('raw', 3)]:.1f}\\%"),
    ("2x2 PhysPrior cells (re-run, float removed)",
     f"& {X2[('physprior', 2)]:.1f}\\% & {X2[('physprior', 3)]:.1f}\\%"),
    # agreement block (float -> prose): pre-fix counts
    ("agreement share (pre-fix)", r"$80.20\%$"),
    ("agreement share prose (pre-fix)", r"on $80.20\%$ of the $197$ pixels"),
    ("agreement table row (pre-fix)", r"158/197 (80.20\%)"),
    ("kappa (pre-fix)", r"$+0.604$"),
    ("kappa prose (pre-fix)", r"\kappa = +0.604"),
    ("McNemar counts (pre-fix)", r"$(25, 14)$"),
    ("McNemar p (pre-fix)", r"$0.108$"),
    ("both correct (pre-fix)", r"$117$ both correct"),
    ("rule only (pre-fix)", r"$20$ rule only"),
    ("DL only (pre-fix)", r"$19$ DL only"),
    # per-station (float -> prose): pre-fix recall / specificity
    ("Longmen recall+spec (pre-fix)", r"77.8\% & 71.5\%"),
    ("combined recall+spec (pre-fix)", r"82.8\% & 63.3\%"),
    ("Longmen recall prose (pre-fix)", r"recall $77.8\%$"),
    ("combined recall prose (pre-fix)", r"recall $82.8\%$"),
    ("Longmen specificity prose (pre-fix)", r"specificity $71.5\%$"),
    ("combined specificity prose (pre-fix)", r"specificity $63.3\%$"),
    # per-season (float -> prose): pre-fix summer / autumn values
    ("summer accuracy (pre-fix table)", r"& 69.4\% & [58.3, 79.2]"),
    ("autumn accuracy (pre-fix table)", r"& 80.0\% & [67.5, 92.5]"),
    ("summer accuracy (pre-fix)", r"$69.4\%$"),
    ("autumn accuracy (pre-fix)", r"$80.0\%$"),
    ("summer prose (pre-fix)", r"$69.4\%$ in summer"),
    ("autumn prose (pre-fix)", r"$80.0\%$ in autumn"),
    ("seasonal range claim (pre-fix)", r"$44.8\%$ winter to $80.0\%$ autumn"),
]:
    stale(_label, _s)


# ---- stale values that a partial update would leave behind ------------------
STALE_SUFFIX = " (stale value from the pre-determinism run)"


def build_stale():
    """Values that must not survive the re-run are supplied on the command line."""
    for arg in sys.argv[1:]:
        label, _, value = arg.partition("=")
        stale(label, value)


build_stale()

# ---- run -------------------------------------------------------------------
fails = []
print("=" * 78)
print(f"manuscript number gate   tag={TAG}   {os.path.basename(TEX)}")
print("=" * 78)

for label, s in CHECKS:
    ok = re.sub(r"[ \t]+", " ", s) in tex_norm
    print(f"  [{'ok ' if ok else 'FAIL'}] {label:<44} {s}")
    if not ok:
        fails.append(label)

if INNER:
    print("\n  -- artifact-internal consistency (no tex involved) --")
    for label, ok in INNER:
        print(f"  [{'ok ' if ok else 'FAIL'}] {label:<44}")
        if not ok:
            fails.append(f"consistency: {label}")

if STALE:
    print("\n  -- values that must not appear --")
    for label, s in STALE:
        # matched against the normalised text, so a superseded number cannot
        # survive a reflow of the table's alignment padding
        present = re.sub(r"[ \t]+", " ", s) in tex_norm
        print(f"  [{'FAIL' if present else 'ok '}] {label:<44} {s}")
        if present:
            fails.append(f"stale: {label}")

if NOT_IN_TEX:
    print("\n  -- NOT ASSERTED: values the revision removed from the manuscript --")
    for label, s in NOT_IN_TEX:
        print(f"  [----] {label:<44} {s}")

print("\n" + "=" * 78)
print(f"checked {len(CHECKS)} expected values, {len(INNER)} consistency guards, "
      f"{len(STALE)} forbidden values, {len(NOT_IN_TEX)} values no longer stated in the tex")
if fails:
    print(f"RESULT: {len(fails)} FAILURES")
    for f in fails:
        print(f"  - {f}")
else:
    print("RESULT: all manuscript numbers trace to the current artifacts")
print("=" * 78)
sys.exit(1 if fails else 0)
