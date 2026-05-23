import re
path = r"E:\Claude code\paper\noise_label_grsl\manuscript_grsl.tex"
with open(path, "r", encoding="utf-8") as f:
    t = f.read()

fixes = [
    ("Training MT-UNet on these", "Training a deep learning segmentation model on these"),
    ("The MT-UNet model  includes", "Standard deep learning segmentation models include"),
    ("The downstream MT-UNet model accuracy", "The downstream model accuracy"),
    ("the downstream MT-UNet model accuracy", "the downstream model accuracy"),
    ("MT-UNet-based", "model-based"),
]
for old, new in fixes:
    if old in t:
        t = t.replace(old, new)
        print(f"Fixed: {old[:50]}")

with open(path, "w", encoding="utf-8") as f:
    f.write(t)
print("Done")
