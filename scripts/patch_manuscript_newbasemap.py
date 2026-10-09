"""Sync the manuscript to the run made with the rebuilt basemaps.

Two changes moved the numbers:

  * the basemaps were rebuilt from train+val clear scenes only (S3), which
    changes the third input channel and therefore every 3-channel result;
  * `freeze_predictions_grp.py` now forms the Co-Teaching ensemble the way
    `train_physprior_coteaching_long.py` does -- the two logit fields averaged
    -- instead of OR-ing two binary predictions. That row moves 61.9 -> 76.7%.

The direct-rule rows and the two 2-channel ablation cells do not use the
basemap and are unchanged, which is itself a check that the loader change
introduced nothing of its own.

Run:  python -X utf8 scripts/patch_manuscript_newbasemap.py
Then: python -X utf8 scripts/verify_manuscript_numbers.py  and recompile.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = [
    # ------------------------------------------------------------- abstract
    (r"reach $70.36\% \pm 2.28$~percentage points (pp) accuracy, versus $74.52\% \pm 2.43$~pp when the same correction is combined with Co-Teaching; both are far above the $29.4$--$50.3\%$ reached",
     r"reach $70.66\% \pm 3.60$~percentage points (pp) accuracy, versus $73.81\% \pm 1.38$~pp when the same correction is combined with Co-Teaching; both are far above the $29.4$--$58.4\%$ reached",
     "abstract: five-seed block"),

    (r"corrected labels improve downstream accuracy by $19.8$~pp under both two and three input channels, while the basemap channel costs $7.6$~pp under both label regimes.",
     r"corrected labels improve downstream accuracy by $19.8$ and $19.3$~pp with two and three input channels, while the basemap channel costs $4.6$ and $5.1$~pp.",
     "abstract: ablation deltas"),

    # ------------------------------------------------------------- Sec. I contribution 4
    (r"A model trained on PhysPrior-corrected labels achieves $70.36\% \pm 2.28$~pp accuracy across five seeds, against $41.1\%$ on raw CLDMSK labels and $29.4$--$50.3\%$ for every noise-robust training method tested. In a single-seed $2\times 2$ ablation the two factors are additive: labels are worth $+19.8$~pp and the basemap costs $7.6$~pp, under both channel configurations; repeated-seed and grouped controls remain necessary.",
     r"A model trained on PhysPrior-corrected labels achieves $70.66\% \pm 3.60$~pp accuracy across five seeds, against $46.7\%$ on raw CLDMSK labels and $29.4$--$58.4\%$ for every noise-robust training method tested, so every corrected-label configuration exceeds every raw-label one. In a single-seed $2\times 2$ ablation the two factors are additive: labels are worth $+19.3$~pp at three channels and the basemap costs $5.1$~pp; repeated-seed and grouped controls remain necessary.",
     "Sec. I contribution 4"),

    # ------------------------------------------------------------- Table I rows
    (r"Standard training & 197 & 41.1\%\,[34.5,\,47.7] & 58.3 & 50.0 & +0.235 \\",
     r"Standard training & 197 & 46.7\%\,[39.6,\,53.8] & 61.2 & 51.6 & +0.259 \\",
     "Table I: standard training"),
    (r"Loss Correction & 197 & 50.3\%\,[43.1,\,57.4] & 64.2 & 53.8 & +0.314 \\",
     r"Loss Correction & 197 & 58.4\%\,[51.8,\,65.5] & 70.0 & 58.2 & +0.398 \\",
     "Table I: loss correction"),
    (r"Co-Teaching A & 197 & 39.6\%\,[33.0,\,46.7] & 56.2 & 48.5 & +0.172 \\",
     r"Co-Teaching A & 197 & 48.7\%\,[41.6,\,55.8] & 63.2 & 53.0 & +0.298 \\",
     "Table I: co-teaching A"),
    (r"Co-Teaching B & 197 & 38.6\%\,[32.0,\,45.2] & 55.5 & 48.1 & +0.158 \\",
     r"Co-Teaching B & 197 & 51.8\%\,[44.7,\,58.9] & 65.3 & 54.5 & +0.330 \\",
     "Table I: co-teaching B"),
    (r"GCE ($q{=}0.7$) & 197 & 39.6\%\,[33.0,\,46.7] & 56.2 & 48.5 & +0.172 \\",
     r"GCE ($q{=}0.7$) & 197 & 44.7\%\,[38.1,\,51.8] & 59.8 & 50.7 & +0.236 \\",
     "Table I: GCE"),
    (r"PhysPrior moderate + DL (3 channels) & 197 & \textbf{71.6\%}\,[65.0,\,77.7] & \textbf{72.3} & \textbf{60.6} & \textbf{+0.411} \\",
     r"PhysPrior moderate + DL (3 channels) & 197 & 69.0\%\,[62.4,\,75.6] & 73.0 & 61.1 & +0.420 \\",
     "Table I: PhysPrior + DL"),
    (r"PhysPrior + Co-Teaching (predef.\ ensemble) & 197 & 61.9\%\,[55.3,\,68.5] & 68.0 & 56.1 & +0.332 \\",
     r"PhysPrior + Co-Teaching (predef.\ ensemble) & 197 & \textbf{76.7\%}\,[70.6,\,82.2] & \textbf{75.4} & \textbf{64.6} & \textbf{+0.481} \\",
     "Table I: PhysPrior + Co-Teaching"),

    # ------------------------------------------------------------- Sec. III-B
    (r"the same architecture reaches $71.6\%$ on PhysPrior-corrected labels against $41.1\%$ on raw labels, while loss correction, Co-Teaching, GCE and Mixup on raw labels reach only $29.4$--$50.3\%$.",
     r"the same architecture reaches $69.0\%$ on PhysPrior-corrected labels against $46.7\%$ on raw labels, while loss correction, Co-Teaching, GCE and Mixup on raw labels reach only $29.4$--$58.4\%$, so the corrected-label block and the raw-label block do not overlap.",
     "Sec. III-B: label vs method"),
    (r"as a single-seed $2\times 2$ cell under three channels ($65.0\%$ on a fixed $20$-epoch budget, Section~\ref{sec:abl}), and as the five-seed DL mean $70.36\%\pm 2.28$~pp",
     r"as a single-seed $2\times 2$ cell under three channels ($67.5\%$ on a fixed $20$-epoch budget, Section~\ref{sec:abl}), and as the five-seed DL mean $70.66\%\pm 3.60$~pp",
     "Sec. III-B: PhysPrior appearances"),

    # ------------------------------------------------------------- Table IV (station)
    (r"Changsha (urban) & 29  & 62.1\% & 60.0\% & 46.2\% & 75.0\% \\",
     r"Changsha (urban) & 29  & 44.8\% & 44.8\% & 100.0\% & 0.0\% \\",
     "Table III: Changsha row"),
    (r"Longmen (suburban) & 168 & 73.2\% & 50.0\% & 82.2\% & 69.9\% \\",
     r"Longmen (suburban) & 168 & 73.2\% & 50.0\% & 77.8\% & 71.5\% \\",
     "Table III: Longmen row"),
    (r"Combined & 197 & 71.6\% & 51.2\% & 74.1\% & 70.5\% \\",
     r"Combined & 197 & 69.0\% & 48.5\% & 82.8\% & 63.3\% \\",
     "Table III: combined row"),

    # ------------------------------------------------------------- Table IV-b (season)
    (r"Spring (Mar--May) & 71.4\% & [58.9, 82.1] & 56 \\",
     r"Spring (Mar--May) & 73.2\% & [60.7, 83.9] & 56 \\",
     "Table IV-b: spring"),
    (r"Summer (Jun--Aug) & 69.4\% & [58.3, 80.6] & 72 \\",
     r"Summer (Jun--Aug) & 69.4\% & [58.3, 79.2] & 72 \\",
     "Table IV-b: summer"),
    (r"Autumn (Sep--Nov) & 82.5\% & [70.0, 92.5] & 40 \\",
     r"Autumn (Sep--Nov) & 80.0\% & [67.5, 92.5] & 40 \\",
     "Table IV-b: autumn"),
    (r"Winter (Dec--Feb) & 62.1\% & [44.8, 79.3] & 29 \\",
     r"Winter (Dec--Feb) & 44.8\% & [27.6, 62.1] & 29 \\",
     "Table IV-b: winter"),

    # ------------------------------------------------------------- Table V (2x2)
    (r"Raw CLDMSK            & 52.8\%  & 45.2\% \\",
     r"Raw CLDMSK            & 52.8\%  & 48.2\% \\",
     "Table V: raw cells"),
    (r"PhysPrior moderate    & \textbf{72.6\%}  & 65.0\% \\",
     r"PhysPrior moderate    & \textbf{72.6\%}  & 67.5\% \\",
     "Table V: PhysPrior cells"),
    (r"\quad Under Raw CLDMSK  & \multicolumn{2}{r}{$-7.61$~pp} \\",
     r"\quad Under Raw CLDMSK  & \multicolumn{2}{r}{$-4.57$~pp} \\",
     "Table V: basemap effect under raw"),
    (r"\quad Under PhysPrior   & \multicolumn{2}{r}{$-7.61$~pp} \\",
     r"\quad Under PhysPrior   & \multicolumn{2}{r}{$-5.08$~pp} \\",
     "Table V: basemap effect under PhysPrior"),
    (r"\quad Under 3-ch        & \multicolumn{2}{r}{$+19.80$~pp} \\",
     r"\quad Under 3-ch        & \multicolumn{2}{r}{$+19.29$~pp} \\",
     "Table V: PhysPrior effect 3ch"),
    (r"The single-seed $2 \times 2$ ablation separates into two additive main effects with no interaction: correcting the labels is worth $+19.8$~pp in both channel configurations, and adding the basemap costs $7.6$~pp under both label regimes.",
     r"The single-seed $2 \times 2$ ablation separates into two additive main effects with no interaction: correcting the labels is worth $+19.8$~pp with two channels and $+19.3$~pp with three, and adding the basemap costs $4.6$--$5.1$~pp under both label regimes.",
     "Table V narrative"),

    # ------------------------------------------------------------- Table VI
    (r"Downstream DL correct against radar                & 141/197 (71.57\%) \\",
     r"Downstream DL correct against radar                & 136/197 (69.04\%) \\",
     "Table VI: DL correct"),
    (r"\emph{Both} correct against radar                   & 120/197 (60.91\%) \\",
     r"\emph{Both} correct against radar                   & 117/197 (59.39\%) \\",
     "Table VI: both correct"),
    (r"Direct PhysPrior only correct                     & 17/197 \\",
     r"Direct PhysPrior only correct                     & 20/197 \\",
     "Table VI: rule only"),
    (r"Downstream DL only correct                        & 21/197 \\",
     r"Downstream DL only correct                        & 19/197 \\",
     "Table VI: DL only"),
    (r"Both incorrect                                    & 39/197 \\",
     r"Both incorrect                                    & 41/197 \\",
     "Table VI: both incorrect"),
    (r"Sample-wise agreement (label $=$ model)           & 159/197 (80.71\%) \\",
     r"Sample-wise agreement (label $=$ model)           & 158/197 (80.20\%) \\",
     "Table VI: agreement"),
    (r"Cohen's $\kappa$                                   & $+0.608$ \\",
     r"Cohen's $\kappa$                                   & $+0.604$ \\",
     "Table VI: kappa"),
    (r"McNemar discordant counts $(b,c)$                 & $(17, 21)$ \\",
     r"McNemar discordant counts $(b,c)$                 & $(25, 14)$ \\",
     "Table VI: McNemar counts"),
    (r"Exact McNemar $p$-value (two-sided)               & $0.627$ \\",
     r"Exact McNemar $p$-value (two-sided)               & $0.108$ \\",
     "Table VI: McNemar p"),
    (r"Five-seed mean accuracy (PhysPrior$+$DL)             & $70.36\% \pm 2.28$~pp \\",
     r"Five-seed mean accuracy (PhysPrior$+$DL)             & $70.66\% \pm 3.60$~pp \\",
     "Table VI: five-seed accuracy"),
    (r"Five-seed mean balanced accuracy (PhysPrior$+$DL)         & $72.16\% \pm 1.19$~pp \\",
     r"Five-seed mean balanced accuracy (PhysPrior$+$DL)         & $73.88\% \pm 2.54$~pp \\",
     "Table VI: five-seed BA"),
    (r"Five-seed mean F1 (PhysPrior$+$DL)                       & $60.26\% \pm 1.22$~pp \\",
     r"Five-seed mean F1 (PhysPrior$+$DL)                       & $62.22\% \pm 2.84$~pp \\",
     "Table VI: five-seed F1"),
    (r"Five-seed mean recall (PhysPrior$+$DL)                  & $76.55\% \pm 7.76$~pp \\",
     r"Five-seed mean recall (PhysPrior$+$DL)                  & $81.72\% \pm 4.31$~pp \\",
     "Table VI: five-seed recall"),
    (r"Five-seed mean specificity (PhysPrior$+$DL)             & $67.77\% \pm 6.25$~pp \\",
     r"Five-seed mean specificity (PhysPrior$+$DL)             & $66.04\% \pm 5.82$~pp \\",
     "Table VI: five-seed specificity"),

    # ------------------------------------------------------------- Sec. III-G narrative
    (r"The downstream model agrees with the direct rule on $80.71\%$ of the $197$ pixels ($\kappa = +0.608$, McNemar $b = 17$, $c = 21$, $p = 0.627$; $120$ both correct, $17$ rule only, $21$ DL only), so it is not a copy of the rule; on the Changsha subset it under-predicts cloud ($46.2\%$ recall, $75.0\%$ specificity).",
     r"The downstream model agrees with the direct rule on $80.20\%$ of the $197$ pixels ($\kappa = +0.604$, McNemar $b = 25$, $c = 14$, $p = 0.108$; $117$ both correct, $20$ rule only, $19$ DL only), so it is not a copy of the rule; on the Changsha subset it predicts cloud for almost every patch ($100\%$ recall, $0.0\%$ specificity).",
     "Sec. III-G agreement narrative"),

    (r"The five-seed PhysPrior$+$DL mean is $70.36\% \pm 2.28$~pp accuracy (range $66.50$--$73.10$\%), $26.2$~pp above the raw-CLDMSK baseline. It is indistinguishable from the always-clear $70.6\%$ in accuracy alone, but that classifier attains the figure with $0\%$ cloud recall against the five-seed mean recall of $76.6\%$. Adding Co-Teaching raises the five-seed mean to $74.52\% \pm 2.43$~pp. A paired-seed PhysPrior-versus-raw comparison remains the natural next step.",
     r"The five-seed PhysPrior$+$DL mean is $70.66\% \pm 3.60$~pp accuracy (range $65.48$--$75.63$\%), $26.5$~pp above the raw-CLDMSK baseline. It is indistinguishable from the always-clear $70.6\%$ in accuracy alone, but that classifier attains the figure with $0\%$ cloud recall against the five-seed mean recall of $81.7\%$. Adding Co-Teaching raises the five-seed mean to $73.81\% \pm 1.38$~pp. A paired-seed PhysPrior-versus-raw comparison remains the natural next step.",
     "Sec. III-G five-seed narrative"),

    # ------------------------------------------------------------- Sec. IV
    (r"Seasonal accuracy ranges from $62.1\%$ winter to $82.5\%$ autumn (Table~\ref{tab:season}), but the $29$ winter samples are all Changsha (2022, 2024 and 2025) and the $168$ non-winter all Longmen (2020): station, year, season and setting are mutually confounded. Within Longmen alone, where site is fixed, the spread persists---$69.4\%$ in summer ($n{=}72$) against $82.5\%$ in autumn ($n{=}40$)---so it is not only a station contrast; all $168$ Longmen patches are from one year, so season and year remain inseparable.",
     r"Seasonal accuracy ranges from $44.8\%$ winter to $80.0\%$ autumn (Table~\ref{tab:season}), but the $29$ winter samples are all Changsha (2022, 2024 and 2025) and the $168$ non-winter all Longmen (2020): station, year, season and setting are mutually confounded. Within Longmen alone, where site is fixed, the spread persists---$69.4\%$ in summer ($n{=}72$) against $80.0\%$ in autumn ($n{=}40$)---so it is not only a station contrast; all $168$ Longmen patches are from one year, so season and year remain inseparable.",
     "Sec. IV-B season narrative"),

    (r"radar agreement rises from $44.2\%$ to $69.5\%$, and training on the corrected labels lifts the same architecture from $41.1\%$ to $71.6\%$. Both M15-only rows nevertheless attain higher raw accuracy ($76.6\%$ and $77.2\%$), so PhysPrior is a label-bias correction framework, not a substitute for a well-tuned M15 threshold.",
     r"radar agreement rises from $44.2\%$ to $69.5\%$, and training on the corrected labels lifts the same architecture from $46.7\%$ to $69.0\%$. The matched M15-only row ties the best corrected-label configuration ($76.6\%$) and the post-hoc oracle exceeds it ($77.2\%$), so PhysPrior is a label-bias correction framework, not a substitute for a well-tuned M15 threshold.",
     "Sec. IV-A"),

    # ------------------------------------------------------------- Sec. V
    (r"models trained on the corrected labels reach $70.36\% \pm 2.28$~pp across five seeds, against $41.1\%$ on raw labels and $29.4$--$50.3\%$ for four noise-robust methods. It remains a label-bias correction framework rather than an operational replacement: the matched M15 threshold reaches $76.6\%$, and Co-Teaching adds $4.16$~pp ($74.52\% \pm 2.43$~pp).",
     r"models trained on the corrected labels reach $70.66\% \pm 3.60$~pp across five seeds, against $46.7\%$ on raw labels and $29.4$--$58.4\%$ for four noise-robust methods. It remains a label-bias correction framework rather than an operational replacement: the matched M15 threshold reaches $76.6\%$, and Co-Teaching adds $3.15$~pp ($73.81\% \pm 1.38$~pp).",
     "Sec. V conclusion"),
]


def main():
    s = open(TEX, encoding="utf-8").read()
    problems = []
    for old, new, label in R:
        n = s.count(old)
        if n != 1:
            problems.append(f"  {label}: found {n} times (need exactly 1)")
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
