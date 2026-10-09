"""Update Table V and its prose to the fixed-budget 2x2 ablation.

The ablation originally selected the best-validation checkpoint. That criterion
is measured on the 28 radar-labelled validation patches, where an always-cloud
prediction already scores 17/28, and it selected epoch 2, 1, 1 and 12 for the
four cells — three of them at the trivial point. The ablation now trains all four
cells for a fixed 20 epochs with no early stopping and reports the final model,
which removes the selector from the comparison.

Run:  python -X utf8 scripts/patch_manuscript_ablation.py
Then: python -X utf8 scripts/verify_manuscript_numbers.py
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = [
    # ---- protocol description -------------------------------------------
    (r"Each configuration was trained for up to $20$ epochs with early stopping.",
     r"Each configuration was trained for a fixed $20$ epochs with no early stopping, so the four cells share one budget and no checkpoint selection enters the comparison.",
     "ablation protocol sentence"),

    (r"All four configurations share identical hyperparameters, training budget and seed ($42$), and are evaluated on the same $n{=}197$ test set. Deltas are computed from the full-precision table builder output, not from the one-decimal figures shown in the table; the underlying numbers therefore differ by up to $\pm 0.05$~pp from the displayed deltas.",
     r"All four configurations share identical hyperparameters, a fixed $20$-epoch budget with no early stopping, and seed ($42$), and are evaluated on the same $n{=}197$ test set. Deltas use full-precision values, so the displayed one-decimal figures differ by up to $\pm 0.05$~pp from them.",
     "Table V caption"),

    # ---- table cells -----------------------------------------------------
    (r"Raw CLDMSK            & 49.7\%  & 35.0\% \\",
     r"Raw CLDMSK            & 52.8\%  & 45.2\% \\",
     "Table V: raw cells"),
    (r"PhysPrior moderate    & 62.4\%  & \textbf{71.6\%} \\",
     r"PhysPrior moderate    & \textbf{72.6\%}  & 65.0\% \\",
     "Table V: PhysPrior cells"),
    (r"\quad Under Raw CLDMSK  & \multicolumn{2}{r}{$-14.72$~pp} \\",
     r"\quad Under Raw CLDMSK  & \multicolumn{2}{r}{$-7.61$~pp} \\",
     "Table V: basemap effect under raw"),
    (r"\quad Under PhysPrior   & \multicolumn{2}{r}{$+9.14$~pp} \\",
     r"\quad Under PhysPrior   & \multicolumn{2}{r}{$-7.61$~pp} \\",
     "Table V: basemap effect under PhysPrior"),
    (r"\quad Under 2-ch        & \multicolumn{2}{r}{$+12.69$~pp} \\",
     r"\quad Under 2-ch        & \multicolumn{2}{r}{$+19.80$~pp} \\",
     "Table V: PhysPrior effect 2ch"),
    (r"\quad Under 3-ch        & \multicolumn{2}{r}{$+36.55$~pp} \\",
     r"\quad Under 3-ch        & \multicolumn{2}{r}{$+19.80$~pp} \\",
     "Table V: PhysPrior effect 3ch"),

    # ---- ablation narrative ---------------------------------------------
    (r"The single-seed $2 \times 2$ ablation shows two patterns. First, PhysPrior label correction improves downstream accuracy under both channel configurations ($+12.7$~pp at 2-channel and $+36.5$~pp at 3-channel); on this training set the choice of labels matters more than the choice of channels under both label regimes. Second, the basemap channel behaves differently under the two label regimes: it costs $14.7$~pp under raw labels but gains $9.1$~pp under corrected labels. That interaction neither establishes nor excludes shortcut use, because in-distribution accuracy can move either way when geographically specific correlations are exploited. Cross-site and repeated-seed controls remain necessary.",
     r"The single-seed $2 \times 2$ ablation separates into two additive main effects with no interaction: correcting the labels is worth $+19.8$~pp in both channel configurations, and adding the basemap costs $7.6$~pp under both label regimes. On this training set the choice of labels therefore dominates the choice of channels. The basemap effect is negative in both regimes, so it neither establishes nor excludes shortcut use: in-distribution accuracy can move either way when geographically specific correlations are exploited, and the permutation control that would settle it is listed in Section~\ref{sec:limit}.",
     "ablation narrative"),

    # ---- abstract --------------------------------------------------------
    (r"corrected labels improve downstream accuracy by $12.7$~pp with two input channels and $36.5$~pp with three.",
     r"corrected labels improve downstream accuracy by $19.8$~pp under both two and three input channels, while the basemap channel costs $7.6$~pp under both label regimes.",
     "abstract: ablation result"),

    # ---- introduction ----------------------------------------------------
    (r"PhysPrior improves downstream accuracy under both channel configurations, by $12.7$~pp with two channels and $36.5$~pp with three channels;",
     r"PhysPrior improves downstream accuracy additively under both channel configurations, by $+19.8$~pp, while the basemap channel costs $7.6$~pp;",
     "intro: contribution 4 ablation"),

    # ---- methods compared ------------------------------------------------
    (r"as a single-seed $2\times 2$ cell under three channels ($71.6\%$, Section~\ref{sec:abl})",
     r"as a single-seed $2\times 2$ cell under three channels ($65.0\%$ on a fixed $20$-epoch budget, Section~\ref{sec:abl})",
     "text before Table I: 2x2 cell value"),
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
