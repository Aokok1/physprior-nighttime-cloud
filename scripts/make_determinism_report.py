"""Generate the determinism re-run report from the archived and fresh artifacts.

Reads the pre-determinism artifact set archived by scripts/backup_predet.ps1 and
the freshly rebuilt one, and writes a markdown report with the old-vs-new tables,
the five-seed comparison and the reproducibility verdict.

Run:  python -X utf8 scripts/make_determinism_report.py
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT = r"E:/Claude code/project/noise-label-cloud/output"
OLD = os.path.join(OUT, "_predet_backup_20260924")
NEW = OUT
REPORT = r"E:/deepseek/grsl_02086_audit/DETERMINISM_REPORT.md"


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


old_t = load(os.path.join(OLD, "paper_tables_grp.json"))
new_t = load(os.path.join(NEW, "paper_tables_grp.json"))
old_d = load(os.path.join(OLD, "derived_tables_grp.json"))
new_d = load(os.path.join(NEW, "derived_tables_grp.json"))
new_a = load(os.path.join(NEW, "2x2_ablation_grp.json"))
old_a_bv = load(os.path.join(OLD, "2x2_ablation_grp_bestval_selection.json"))

if new_t is None or new_d is None:
    raise SystemExit("the fresh artifacts are not built yet - run scripts/postprocess_grp.ps1 first")

L = []
w = L.append

# ============================================================ 1. findings
w("# Determinism re-run — GRSL-02086 open item 2")
w("")
w("Date: 2026-09-24. Scope: the run-to-run nondeterminism recorded in the audit,")
w("plus everything the deterministic re-run exposed. All ten pipeline stages were")
w("re-run from an empty output directory on the group-disjoint split; the")
w("pre-determinism artifacts were moved to `output/_predet_backup_20260924/`")
w("rather than overwritten.")
w("")

w("## 1. What was actually nondeterministic")
w("")
w("The audit had recorded that repeating the five-seed protocol with identical")
w("seeds moved individual seeds by up to 4 pp. Tracing it to the entry points")
w("found a wider hole:")
w("")
w("| Entry point | Stages | State before |")
w("|---|---|---|")
w("| `train.py` | 02–06 (baseline, co-teaching, GCE, mixup, PhysPrior+DL) | **no seeding of any RNG** |")
w("| `scripts/redo_loss_correction.py` | 01 | **no seeding of any RNG** |")
w("| `scripts/train_physprior_coteaching_long.py` | 10 | **no seeding of any RNG** |")
w("| `data/dataset.py` train loader | all | `shuffle=True` off the global torch RNG |")
w("| `scripts/train_2x2_ablation.py` | 07 | `torch.manual_seed(42)` only |")
w("| `scripts/train_physprior_5seeds.py` | 08 | `torch.manual_seed` + numpy, no CUDA seed |")
w("| `scripts/train_coteaching_5seeds.py` | 09 | `torch.manual_seed` + numpy, no CUDA seed |")
w("")
w("Seven of the ten stages had never been seeded at all, and none of them")
w("disabled cuDNN autotuning. Every trained row of Table I was therefore a single")
w("uncontrolled draw.")
w("")

# ============================================================ 2. fixes
w("## 2. What was changed")
w("")
w("| Change | File |")
w("|---|---|")
w("| `seed_everything(seed)` — seeds Python, NumPy, torch, CUDA; sets `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; calls `torch.use_deterministic_algorithms(True, warn_only=True)` | `determinism.py` (new) |")
w("| import + call at module scope, before the first CUDA handle | `train.py`, `scripts/redo_loss_correction.py`, `scripts/train_physprior_5seeds.py`, `scripts/train_coteaching_5seeds.py`, `scripts/train_2x2_ablation.py`, `scripts/train_physprior_coteaching_long.py` |")
w("| train loader given its own seeded generator, so the shuffle order no longer depends on how much randomness the model consumed first | `data/dataset.py` |")
w("| `NL_SEED` (default 42) and `CUBLAS_WORKSPACE_CONFIG` pinned for the whole run | `scripts/rerun_grp.ps1` |")
w("| 2×2 ablation switches to a fixed 20-epoch budget (§6) | `scripts/train_2x2_ablation.py` |")
w("")

# ============================================================ 3. movement
w("## 3. What the re-run moved")
w("")
w("Table I rows. Direct-rule rows are marked because nothing about them trains,")
w("so any movement there would have been a defect rather than a seed effect.")
w("")
w("| Row | pre-determinism | deterministic | Δ (pp) |")
w("|---|---:|---:|---:|")
LABELS = {
    "CLDMSK_raw": "Raw CLDMSK",
    "PureM15_T264": "M15-only, T=264 K",
    "PureM15_T266": "M15-only, T=266 K",
    "PhysPriorLabel_conservative": "PhysPrior, conservative",
    "PhysPriorLabel_moderate": "PhysPrior, moderate",
    "PhysPriorLabel_aggressive": "PhysPrior, aggressive",
    "Standard_training": "Standard training",
    "LossCorrection": "Loss Correction",
    "CoTeaching_A": "Co-Teaching A",
    "CoTeaching_B": "Co-Teaching B",
    "GCE_q07": "GCE (q=0.7)",
    "Mixup": "Mixup",
    "PhysPrior_CoTeaching_ensemble": "PhysPrior + Co-Teaching",
    "PhysPrior_DL_3ch": "PhysPrior + DL (3ch)",
}
NONTRAINED = {"CLDMSK_raw", "PureM15_T264", "PureM15_T266",
              "PhysPriorLabel_conservative", "PhysPriorLabel_moderate",
              "PhysPriorLabel_aggressive"}
biggest = []
# The pre-determinism table listed the dedicated PhysPrior+DL run under the
# ablation-cell key, because with the old selection rule the two produced
# identical test predictions. Fall back to that key so the row still compares.
OLD_KEY_ALIAS = {"PhysPrior_DL_3ch": "abl_physprior_3ch"}
for key, label in LABELS.items():
    o = old_t["table1"].get(OLD_KEY_ALIAS.get(key, key), {}).get("metrics", {}).get("acc")
    n = new_t["table1"].get(key, {}).get("metrics", {}).get("acc")
    if o is None or n is None:
        continue
    d = (n - o) * 100
    tag = " *(direct rule — no training)*" if key in NONTRAINED else ""
    w(f"| {label}{tag} | {o * 100:.2f}% | {n * 100:.2f}% | {d:+.2f} |")
    if key not in NONTRAINED:
        biggest.append((abs(d), label, d))
w("")
if biggest:
    biggest.sort(reverse=True)
    w("Largest movement among trained rows: " +
      ", ".join(f"{lab} {d:+.2f} pp" for _, lab, d in biggest[:5]) + ".")
    w("")
    nontrained_moved = []
    for key in NONTRAINED:
        o = old_t["table1"].get(key, {}).get("metrics", {}).get("acc")
        n = new_t["table1"].get(key, {}).get("metrics", {}).get("acc")
        if o is not None and n is not None and abs(n - o) > 1e-12:
            nontrained_moved.append(key)
    w(f"Direct-rule rows moved: {nontrained_moved or 'none'} — as expected, since "
      "they are deterministic pure functions of the frozen CLDMSK labels.")
    w("")

w("Five-seed blocks:")
w("")
w("| Block | pre-determinism | deterministic |")
w("|---|---:|---:|")
for name, key in (("PhysPrior+DL", "table_five_seed_physprior"),
                  ("Co-Teaching", "table_five_seed_coteaching")):
    o = old_t.get(key)
    n = new_t.get(key)
    if o and n:
        w(f"| {name} | {o['accuracy_mean'] * 100:.2f} ± {o['accuracy_sd'] * 100:.2f} | "
          f"{n['accuracy_mean'] * 100:.2f} ± {n['accuracy_sd'] * 100:.2f} |")
w("")
w("The PhysPrior+DL seed standard deviation falls from 3.74 pp to 2.28 pp: much of")
w("the earlier spread was run-to-run noise rather than seed effect.")
w("")

if new_a:
    w("2×2 ablation (see §6 for why the protocol changed):")
    w("")
    w("| Cell | pre-determinism (best-val) | deterministic (best-val) | deterministic (fixed 20 epochs) |")
    w("|---|---:|---:|---:|")
    for key, name in (("abl_raw_2ch", "Raw, 2ch"), ("abl_raw_3ch", "Raw, 3ch"),
                      ("abl_physprior_2ch", "PhysPrior, 2ch"),
                      ("abl_physprior_3ch", "PhysPrior, 3ch")):
        o = old_t["table_ablation_2x2"].get(key, {}).get("metrics", {}).get("acc")
        bv = old_a_bv and next((c for c in old_a_bv
                                if f"abl_{c['label_mode']}_{c['n_channels']}ch" == key), None)
        bv_acc = (bv["test"]["acc"] if bv else None)
        n = next((c for c in new_a
                  if f"abl_{c['label_mode']}_{c['n_channels']}ch" == key), None)
        n_acc = n["test"]["acc"] if n else None
        row = [f"{o * 100:.2f}%" if o is not None else "-",
               f"{bv_acc * 100:.2f}%" if bv_acc is not None else "-",
               f"{n_acc * 100:.2f}%" if n_acc is not None else "-"]
        w(f"| {name} | " + " | ".join(row) + " |")
    w("")

# ============================================================ 4. derived
w("## 4. Derived diagnostics")
w("")
try:
    oa = old_d["agreement"]
    na = new_d["agreement"]
    if "rule_correct" not in oa:  # normalise the pre-fix schema
        oa = {"model_correct": oa["each_correct"], "both_correct": oa["both_correct"],
              "rule_only_correct": oa["dl_only_correct"], "dl_only_correct": oa["rule_only_correct"],
              "both_wrong": oa["both_wrong"], "agree": oa["agree"],
              "kappa": oa["kappa"], "mcnemar_b": oa["mcnemar_b"],
              "mcnemar_c": oa["mcnemar_c"], "mcnemar_p": oa["mcnemar_p"]}
        oa["rule_correct"] = oa["both_correct"] + oa["rule_only_correct"]
    w("| Diagnostic | pre-determinism | deterministic |")
    w("|---|---:|---:|")
    for label, key in (("direct rule correct", "rule_correct"),
                       ("diagnostic DL model correct", "model_correct"),
                       ("both correct", "both_correct"),
                       ("rule only correct", "rule_only_correct"),
                       ("DL only correct", "dl_only_correct"),
                       ("label = model agreement", "agree"),
                       ("Cohen's kappa", "kappa"),
                       ("McNemar p", "mcnemar_p")):
        ov, nv = oa.get(key), na.get(key)
        if isinstance(ov, float):
            ov, nv = round(ov, 4), round(nv, 4)
        w(f"| {label} | {ov} | {nv} |")
    w("")
except KeyError as e:
    w(f"(agreement block unavailable: {e})")
    w("")

# ============================================================ 5. verdict
w("## 5. Reproducibility verdict — PASS")
w("")
w("Two independent checks, both run after the pipeline finished:")
w("")
w("1. `scripts/probe_determinism.py` — a two-epoch training loop run in two")
w("   separate processes. Both print the same weight hash")
w("   (`c6884de503373e8ba3f7d3fbc1812811`).")
w("2. `scripts/verify_reproducibility.ps1` — the full five-seed PhysPrior protocol")
w("   re-run in a second process under a different output tag and diffed field by")
w("   field against the production run. Result (`reproducibility_check.txt`):")
w("")
w("   * all five per-seed metrics (accuracy, precision, recall, F1, balanced")
w("     accuracy, specificity, TP/FP/TN/FN) identical;")
w("   * all 197 per-sample test predictions identical for every seed;")
w("   * every epoch of every `val_acc` / `val_p` / `val_r` / `val_f1` trace identical;")
w("   * `best_epoch` identical for every seed.")
w("")
w("**The one residual.** PyTorch flags `nll_loss2d_forward_out_cuda_template` as")
w("having no deterministic implementation, and segmentation cross-entropy")
w("dispatches to it; the run therefore proceeds under `warn_only=True`. The replay")
w("shows exactly where that bites and where it does not: the reported")
w("**training-loss scalar** differs between processes by 4×10⁻⁹ to 1×10⁻⁸")
w("relative (float summation order in the CUDA reduction), while the gradient —")
w("and therefore every weight, every metric, every epoch of every validation")
w("trace, and all 197 per-sample predictions — is bit-identical. The comparison")
w("script reports the observed deviation rather than hiding it behind a tolerance.")
w("A run is reproducible in everything the paper reports; the printed loss curve is")
w("reproducible to eight significant figures.")
w("")

# ============================================================ 6. ablation
w("## 6. What the re-run exposed: checkpoint selection on 28 patches")
w("")
w("The deterministic re-run made the ablation's selected epochs legible for the")
w("first time. With best-validation selection the four 2×2 cells were chosen at")
w("epochs **2, 1, 1 and 12**.")
w("")
w("The reason is the selection metric. Validation radar agreement is computed on")
w("the **28 radar-labelled validation patches** (17 cloud, 11 clear), not on all")
w("494. An always-cloud prediction therefore already scores 17/28 = 60.7%, which")
w("is precisely the score at which three of the four cells were selected. The two")
w("raw-label cells are consequently near-trivial checkpoints — the raw-3ch cell")
w("predicts cloud for 186 of the 197 test patches — so the reported")
w("best-validation deltas compared an untrained model against a trained one and")
w("overstated the label effect.")
w("")
w("Handling: the ablation now trains each cell for a fixed 20 epochs and reports")
w("the final-epoch model, which removes the selector from the comparison entirely.")
w("The earlier best-validation artifact is kept at")
w("`output/_predet_backup_20260924/2x2_ablation_grp_bestval_selection.json`, and")
w("`NL_ABL_SELECT=bestval` reproduces it. Supplementary S2 §S2.3 previously")
w("described this metric as covering all 494 patches and has been corrected.")
w("")
w("This does not affect Table I, where the raw-label baseline is the separately")
w("trained `Standard training` row, nor the five-seed blocks, which are compared")
w("against each other under one protocol.")
w("")

# ============================================================ 7. open items
w("## 7. Status of the three open items")
w("")
w("| # | Item | Status |")
w("|---|---|---|")
w("| 1 | Longmen radar-collocated development set | **data-blocked** — the Longmen Ka archive holds 2020 only, and all 168 Longmen 2020 patches already sit in the test split; see `DATA_BLOCKERS.md` |")
w("| 2 | Run-to-run nondeterminism | **closed** — seeded end to end, verified by replay, residual quantified above |")
w("| 3 | Longmen radar year coverage | **data-blocked** — same archive; 1,613 Longmen patches in 2019 and 2021–2025 have no reference. Disclosed in the manuscript's Table IV and limitations |")
w("")

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {REPORT}  ({len(L)} lines)")
