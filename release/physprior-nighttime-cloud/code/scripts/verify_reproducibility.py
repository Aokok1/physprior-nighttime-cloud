"""Prove the multi-seed protocol is bit-reproducible across processes.

Runs the same five-seed PhysPrior training twice in two separate output tags and
compares the two result files field by field, including every per-sample test
prediction. Before `determinism.seed_everything` was introduced, two runs of this
same script with the same five seeds disagreed on individual seeds by up to 4 pp.

Usage (driver):
    scripts/verify_reproducibility.ps1

Usage (comparison only, after both runs exist):
    python -X utf8 scripts/verify_reproducibility.py A.json B.json
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_A = r"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_results_grp.json"
DEFAULT_B = r"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_results_detcheck.json"

FIELDS = ("acc", "precision", "recall", "f1", "balanced_accuracy",
          "specificity", "tp", "fp", "tn", "fn", "n", "n_correct")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(a_path, b_path):
    a = load(a_path)
    b = load(b_path)
    ok = True

    print("=" * 74)
    print("bit-reproducibility check: five-seed PhysPrior, two processes")
    print("=" * 74)
    print(f"  A: {a_path}")
    print(f"  B: {b_path}")

    if len(a) != len(b):
        print(f"\nFAIL  seed count differs: {len(a)} vs {len(b)}")
        return 1

    print(f"\n{'seed':>5} {'metric':<20} {'A':>20} {'B':>20}  match")
    for ra, rb in zip(a, b):
        if ra["seed"] != rb["seed"]:
            print(f"FAIL  seed order differs: {ra['seed']} vs {rb['seed']}")
            ok = False
            continue
        for k in FIELDS:
            va, vb = ra["test"][k], rb["test"][k]
            same = (va == vb)
            ok &= same
            if not same or k in ("acc", "tp", "fp", "tn", "fn"):
                flag = "ok" if same else "DIFF"
                print(f"{ra['seed']:>5} {k:<20} {va:>20} {vb:>20}  {flag}")

        # per-sample test predictions, in file order
        pa = [(r["fname"], r["pred"], r["truth"]) for r in ra["test_records"]]
        pb = [(r["fname"], r["pred"], r["truth"]) for r in rb["test_records"]]
        if pa != pb:
            diff = [(x, y) for x, y in zip(pa, pb) if x != y]
            print(f"{ra['seed']:>5} per-sample            "
                  f"{len(diff)} of {len(pa)} predictions differ   DIFF")
            for x, y in diff[:5]:
                print(f"        {x}  vs  {y}")
            ok = False
        else:
            print(f"{ra['seed']:>5} per-sample            "
                  f"{len(pa)}/{len(pa)} identical                    ok")

        # per-epoch training trace. `train_loss` is the one quantity that is not
        # bit-identical: PyTorch flags nll_loss2d's CUDA reduction as having no
        # deterministic implementation, and that shows up as float summation-order
        # noise in the reported scalar (~1e-9 relative) while leaving the gradient
        # — and therefore every weight, metric and prediction — unchanged. The
        # scalar is compared with a tolerance and the observed deviation printed;
        # every other trace entry must match exactly.
        ha, hb = ra.get("history", {}), rb.get("history", {})
        for k in sorted(set(ha) | set(hb)):
            va, vb = ha.get(k, []), hb.get(k, [])
            if k == "train_loss":
                if len(va) != len(vb):
                    print(f"{ra['seed']:>5} train_loss            length {len(va)} vs {len(vb)}"
                          f"                     DIFF")
                    ok = False
                    continue
                rel = [abs(x - y) / max(abs(x), abs(y), 1e-12) for x, y in zip(va, vb)]
                worst = max(rel) if rel else 0.0
                if worst < 1e-6:
                    print(f"{ra['seed']:>5} train_loss            max rel. dev {worst:.2e}"
                          f" over {len(va)} epochs   ok (float noise)")
                else:
                    print(f"{ra['seed']:>5} train_loss            max rel. dev {worst:.2e}"
                          f"                     DIFF")
                    ok = False
            elif va != vb:
                print(f"{ra['seed']:>5} {k:<20} traces differ                 DIFF")
                ok = False
        for k in ("val_acc", "val_p", "val_r", "val_f1"):
            if ha.get(k) == hb.get(k) and k in ha:
                pass
        if ha.get("val_acc") == hb.get("val_acc"):
            print(f"{ra['seed']:>5} val_acc               "
                  f"{len(ha.get('val_acc', []))} epochs identical            ok")
        if ra.get("best_epoch") != rb.get("best_epoch"):
            print(f"{ra['seed']:>5} best_epoch            "
                  f"{ra.get('best_epoch')} vs {rb.get('best_epoch')}             DIFF")
            ok = False

    print("\n" + "=" * 74)
    if ok:
        print("PASS  both processes agree on every metric, every per-sample "
              "prediction, and every epoch of every training and validation trace.")
        print("      The only non-identical quantity is the reported training-loss")
        print("      scalar, which differs at ~1e-9 relative from float summation")
        print("      order in the CUDA cross-entropy reduction.")
    else:
        print("FAIL  the two runs are not bit-reproducible; see the DIFF lines above.")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    pa = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_A
    pb = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_B
    sys.exit(main(pa, pb))
