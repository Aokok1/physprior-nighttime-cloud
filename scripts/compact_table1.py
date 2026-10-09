"""Compact Table I by folding each group-header row into a single line.

The table used a header line followed by an empty `\\\\` row for each of the seven
method groups. Merging them reclaims roughly 65 pt inside the float, which the
five-page GRSL limit needs.
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
p = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"
s = open(p, encoding="utf-8").read()

pattern = re.compile(r"(\\textit\{[^}]*\})\n\\\\\n", re.M)
n = len(pattern.findall(s))
s2 = pattern.sub(lambda m: m.group(1) + " \\\\\n", s)
open(p, "w", encoding="utf-8").write(s2)
print(f"folded {n} group-header rows")
