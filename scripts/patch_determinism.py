"""Route the three re-run training scripts through the shared determinism helper."""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud/"

TARGETS = [
    "scripts/train_physprior_5seeds.py",
    "scripts/train_coteaching_5seeds.py",
    "scripts/train_2x2_ablation.py",
]

for rel in TARGETS:
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    if "seed_everything" in s:
        print("already patched  " + rel)
        continue

    # replace the hand-rolled seeding blocks with a single call
    s2 = re.sub(
        r"[ \t]*torch\.manual_seed\(([^)]*)\)\n"
        r"(?:[ \t]*np\.random\.seed\([^)]*\)\n)?"
        r"(?:[ \t]*if torch\.cuda\.is_available\(\):\n)?"
        r"(?:[ \t]*torch\.cuda\.manual_seed_all\([^)]*\)\n)?",
        lambda m: f"    seed_everything({m.group(1)})\n",
        s,
    )
    if s2 == s:
        print("no seeding block found in " + rel)
        continue

    # make the helper importable
    if "from determinism import" not in s2:
        anchor = "sys.path.insert(0, r\"E:/Claude code/project/noise-label-cloud\")"
        if anchor in s2:
            s2 = s2.replace(anchor, anchor + "\nfrom determinism import seed_everything",
                            1)
        else:
            lines = s2.split("\n")
            idx = 0
            for i, ln in enumerate(lines[:70]):
                if ln.startswith("import ") or ln.startswith("from "):
                    idx = i
            lines.insert(idx + 1, "from determinism import seed_everything")
            s2 = "\n".join(lines)

    open(p, "w", encoding="utf-8").write(s2)
    print("patched          " + rel)
