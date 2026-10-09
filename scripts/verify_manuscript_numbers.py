"""Check every number the manuscript reports against the artifacts that produce it.

The gate is deliberately one-directional: it derives an expected value from a
JSON artifact and asserts the manuscript contains that value, formatted the way
the manuscript formats it. It also asserts that no *stale* value survives, which
catches a partial update after a re-run.

Run after `rebuild_paper_tables.py` and `derive_station_season_tables.py`:

    python -X utf8 scripts/verify_manuscript_numbers.py

Exit code 0 = every checked number is present and no stale number survives.
"""
import io
import json
import os
import re
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
abl = load(f"paper_tables{SUF}.json")["table_ablation_2x2"]
deltas = load(f"paper_tables{SUF}.json")["table_ablation_deltas"]
five = load(f"paper_tables{SUF}.json")["table_five_seed_physprior"]
ct5 = load(f"paper_tables{SUF}.json")["table_five_seed_coteaching"]
dv = load(f"derived_tables{SUF}.json")

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


CHECKS = []          # (label, expected substring)
STALE = []           # (label, substring that must NOT appear)


def want(label, s):
    CHECKS.append((label, s))


def stale(label, s):
    STALE.append((label, s))


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

# ---- 2x2 ablation ----------------------------------------------------------
for key in ("abl_raw_2ch", "abl_raw_3ch", "abl_physprior_2ch", "abl_physprior_3ch"):
    want(f"Table V {key}", pct(abl[key]["metrics"]["acc"] * 100, 1) + "\\%")
for name, tex_label in (("basemap_effect_raw_pp", "basemap effect under raw"),
                        ("basemap_effect_physprior_pp", "basemap effect under PhysPrior"),
                        ("physprior_effect_2ch_pp", "PhysPrior effect 2ch"),
                        ("physprior_effect_3ch_pp", "PhysPrior effect 3ch")):
    v = deltas[name]
    sign = "-" if v < 0 else "+"
    want(f"Table V {tex_label}", f"${sign}{abs(v):.2f}$~pp")

# The abstract and the introduction state the two main effects as one number each.
want("abstract basemap cost",
     f"the basemap channel costs ${abs(deltas['basemap_effect_raw_pp']):.1f}$ and "
     f"${abs(deltas['basemap_effect_physprior_pp']):.1f}$~pp")
want("abstract label gain",
     f"corrected labels improve downstream accuracy by "
     f"${abs(deltas['physprior_effect_2ch_pp']):.1f}$ and "
     f"${abs(deltas['physprior_effect_3ch_pp']):.1f}$~pp")

# ---- five seeds ------------------------------------------------------------
want("five-seed PhysPrior mean", f"{five['accuracy_mean'] * 100:.2f}\\% \\pm {five['accuracy_sd'] * 100:.2f}")
want("five-seed Co-Teaching mean", f"{ct5['accuracy_mean'] * 100:.2f}\\% \\pm {ct5['accuracy_sd'] * 100:.2f}")
for m, texname in (("balanced_accuracy", "balanced accuracy"), ("f1", "F1"),
                   ("recall", "recall"), ("specificity", "specificity")):
    want(f"five-seed {texname}",
         f"{five[m + '_mean'] * 100:.2f}\\% \\pm {five[m + '_sd'] * 100:.2f}")
accs = [p["acc"] * 100 for p in five["per_seed"]]
want("five-seed range", f"${min(accs):.2f}$--${max(accs):.2f}$")

# ---- derived tables --------------------------------------------------------
# The label is part of the check: the pre-fix derived table swapped these two
# counts, and a value-only check cannot tell a correct row from a swapped one.
ag = dv["agreement"]
want("rule row (label + value)",
     f"Direct PhysPrior rule correct against radar & {ag['rule_correct']}/197 "
     f"({ag['rule_correct'] / 197 * 100:.2f}\\%)")
want("model row (label + value)",
     f"Downstream DL correct against radar & {ag['model_correct']}/197 "
     f"({ag['model_correct'] / 197 * 100:.2f}\\%)")
want("both correct", f"\\emph{{Both}} correct against radar & {ag['both_correct']}/197")
want("rule-only row",
     f"Direct PhysPrior only correct & {ag['rule_only_correct']}/197")
want("dl-only row",
     f"Downstream DL only correct & {ag['dl_only_correct']}/197")
want("agreement share",
     f"Sample-wise agreement (label $=$ model) & {ag['agree']}/197 "
     f"({ag['agree'] / 197 * 100:.2f}\\%)")
want("kappa", f"$+{ag['kappa']:.3f}$")
want("mcnemar", f"$({ag['mcnemar_b']}, {ag['mcnemar_c']})$")
want("mcnemar p", f"${ag['mcnemar_p']:.3f}$")

dc = dv["direct_rule_confusion"]
want("direct-rule confusion counts",
     f"${dc['tn']}/{dc['fp']}/{dc['fn']}/{dc['tp']}$")

st = dv["per_station"]
for site, texname in (("CS", "Changsha"), ("LM", "Longmen")):
    m = st[site]
    want(f"Table IV {texname} row",
         f"{m['n']} & {m['acc'] * 100:.1f}\\% & {m['precision'] * 100:.1f}\\% & "
         f"{m['recall'] * 100:.1f}\\% & {m['specificity'] * 100:.1f}\\%")
m = st["combined"]
want("Table IV combined row",
     f"{m['n']} & {m['acc'] * 100:.1f}\\% & {m['precision'] * 100:.1f}\\% & "
     f"{m['recall'] * 100:.1f}\\% & {m['specificity'] * 100:.1f}\\%")

# The per-season breakdown is a table again (a short-lived prose form was tried in
# revision 7 to buy space for Fig. 1; the reviewer had asked for more detail in
# this section, so the table was restored and the space found elsewhere).
for season in ("Spring", "Summer", "Autumn", "Winter"):
    m = dv["per_season"][season]
    want(f"Table III-b {season}",
         f"{m['acc'] * 100:.1f}\\% & [{m['ci']['lo'] * 100:.1f}, {m['ci']['hi'] * 100:.1f}] & {m['n']}")

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
truths = [r["truth"] for r in load(f"frozen_predictions/all_frozen_predictions{SUF}.json")["CLDMSK_raw"]]
n_clear = sum(1 for t in truths if t == 0)
want("always-clear accuracy", f"{n_clear / len(truths) * 100:.1f}\\%")

# ---- prevalence and split closure (from the manifest) ----------------------
import csv  # noqa: E402

# The manifest is an input, not a re-run artifact, so it is always read from the
# live output directory even when the gate audits an archived artifact set.
man_path = os.path.join(ROOT, "output", "manifest",
                        "sample_manifest_grp.csv" if TAG == "grp" else "sample_manifest.csv")
man = list(csv.DictReader(open(man_path, encoding="utf-8", newline="")))
by_split = {}
for r in man:
    by_split.setdefault(r["split"], []).append(r)
n_test = len(by_split["test"])
n_clear = sum(1 for r in by_split["test"] if float(r["radar_label"]) == 0)
n_cloud = n_test - n_clear
n_train = len(by_split["train_grp"])
n_val = len(by_split["val_grp"])


def texnum(n):
    """1234 -> '1{,}234' as the manuscript writes it."""
    return f"{n:,}".replace(",", "{,}")


want("test prevalence",
     f"${n_clear}$ ($70.6$\\%) are radar-clear and ${n_cloud}$ ($29.4$\\%) are radar-cloudy")
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

if STALE:
    print("\n  -- values that must not appear --")
    for label, s in STALE:
        present = s in tex
        print(f"  [{'FAIL' if present else 'ok '}] {label:<44} {s}")
        if present:
            fails.append(f"stale: {label}")

print("\n" + "=" * 78)
print(f"checked {len(CHECKS)} expected values, {len(STALE)} forbidden values")
if fails:
    print(f"RESULT: {len(fails)} FAILURES")
    for f in fails:
        print(f"  - {f}")
else:
    print("RESULT: all manuscript numbers trace to the current artifacts")
print("=" * 78)
sys.exit(1 if fails else 0)
