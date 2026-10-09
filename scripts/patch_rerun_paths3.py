"""Third-pass patcher: make redo_loss_correction.py read the tagged manifest."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
p = r"E:/Claude code/project/noise-label-cloud/scripts/redo_loss_correction.py"
s = open(p, encoding="utf-8").read()
old = 'df = pd.read_csv(r"E:/Claude code/project/noise-label-cloud/output/manifest/sample_manifest.csv")'
new = ('_mf = ("sample_manifest_grp.csv" if os.environ.get("NL_SPLIT_TAG") == "grp"\n'
       '       else "sample_manifest.csv")\n'
       'df = pd.read_csv(os.path.join(r"E:/Claude code/project/noise-label-cloud/output/manifest", _mf))')
if old in s:
    open(p, "w", encoding="utf-8").write(s.replace(old, new))
    print("patched redo_loss_correction.py")
else:
    print("pattern absent")
