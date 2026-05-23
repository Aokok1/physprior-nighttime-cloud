"""Fix literal \n\n in English manuscript."""
path = r"E:\Claude code\paper\noise_label_grsl\manuscript_grsl.tex"
with open(path, "r", encoding="utf-8") as f:
    t = f.read()

# Replace literal \n\n with actual newlines
t = t.replace("\\n\\n", "\n\n")

with open(path, "w", encoding="utf-8") as f:
    f.write(t)

print("Fixed")
