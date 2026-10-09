"""Define every abbreviation the manuscript uses, in response to R2-7.

The reviewer asked for a pass over unclear abbreviations, naming "pp" as an
example. "pp" is now defined in the abstract; this patch finishes the pass. Each
abbreviation below is expanded or defined at its first occurrence, or in the
table caption that introduces it as a column header.

Run:  python -X utf8 scripts/patch_manuscript_abbreviations.py
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = [
    (r"NASA's VIIRS CLDMSK product is widely used as a global cloud mask",
     r"NASA's VIIRS (Visible Infrared Imaging Radiometer Suite) CLDMSK cloud mask is widely used",
     "VIIRS expanded at first use"),

    (r"only when the M15 brightness temperature exceeds",
     r"only when the M15 (11.45~$\mu$m) brightness temperature exceeds",
     "M15 band defined at first use"),

    (r"The third input channel is the DNB static nighttime-lights basemap.",
     r"The third input channel is the DNB (Day/Night Band) static nighttime-lights basemap.",
     "DNB expanded"),

    (r"where BTD and water-vapour channels could disambiguate",
     r"where brightness-temperature-difference and water-vapour channels could disambiguate",
     "BTD expanded"),

    (r"Bold marks the best value in each metric column.}",
     r"Bold marks the best value in each metric column; BA is balanced accuracy and MCC the Matthews correlation coefficient.}",
     "BA / MCC defined in the Table I caption"),

    (r"where $94/45/15/43$ corresponds to TN/FP/FN/TP.",
     r"where $94/45/15/43$ corresponds to TN/FP/FN/TP (true and false negatives, false and true positives).",
     "TN/FP/FN/TP expanded"),

    (r"\caption{Per-station breakdown of the saved single-checkpoint",
     r"\caption{Per-station breakdown (P: precision, R: recall, Sp.: specificity) of the saved single-checkpoint",
     "P/R/Sp defined in the Table III caption"),
]


def main():
    s = open(TEX, encoding="utf-8").read()
    problems = []
    for old, new, label in R:
        n = s.count(old)
        if n != 1:
            problems.append(f"  {label}: pattern found {n} times (need exactly 1)")
            continue
        s = s.replace(old, new, 1)
    if problems:
        print("ABORTED - no file written:")
        print("\n".join(problems))
        return 1
    open(TEX, "w", encoding="utf-8").write(s)
    print(f"applied {len(R)} replacements to {TEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
