"""Opt the two sample-filtering runs into the five-value MT_Loss return."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud/"

TARGETS = [
    "scripts/train_physprior_5seeds.py",
    "scripts/train_coteaching_5seeds.py",
]

for rel in TARGETS:
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    if "return_per_sample=True" in s:
        print("already set  " + rel)
        continue
    # the single MT_Loss construction in each of these two scripts
    m = re.search(r"MT_Loss\(\s*weight_center=[^)]*?\)", s, re.S)
    if not m:
        print("no MT_Loss(...) found in " + rel)
        continue
    call = m.group(0)
    new = call[:-1].rstrip()
    if not new.endswith(","):
        new += ","
    new += " return_per_sample=True)"
    s = s[:m.start()] + new + s[m.end():]
    open(p, "w", encoding="utf-8").write(s)
    print("patched      " + rel)
    print("   " + " ".join(new.split()))
