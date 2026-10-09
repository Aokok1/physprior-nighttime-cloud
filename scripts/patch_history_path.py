"""Fix the history-path derivation in the two five-seed scripts.

Both scripts derived the per-seed history filename with

    out_path.replace("<name>_results.json", "<name>_history_seed<N>.json")

Once the output path gained a tag suffix (…_results_grp.json) that literal no
longer matched, the replace became a no-op, hist_path collapsed onto out_path,
and the history dump silently overwrote the results file after every seed — so
only the last seed survived and the per-sample records were lost.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud/"

FIXES = [
    ("scripts/train_physprior_5seeds.py",
     'hist_path = out_path.replace("physprior_5seed_results.json", f"physprior_5seed_history_seed{seed}.json")',
     'hist_path = out_path.replace("physprior_5seed_results", f"physprior_5seed_history_seed{seed}")'),
    ("scripts/train_coteaching_5seeds.py",
     'hist_path = out_path.replace("coteaching_5seed_results.json", f"coteaching_5seed_history_seed{seed}.json")',
     'hist_path = out_path.replace("coteaching_5seed_results", f"coteaching_5seed_history_seed{seed}")'),
]

for rel, old, new in FIXES:
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    if new in s:
        print("already fixed  " + rel)
        continue
    if old not in s:
        print("pattern absent " + rel)
        continue
    open(p, "w", encoding="utf-8").write(s.replace(old, new))
    print("fixed          " + rel)
