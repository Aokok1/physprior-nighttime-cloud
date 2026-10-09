"""Update the manuscript to the deterministic re-run's numbers.

Every replacement is an exact literal and must match the file once; the script
aborts on a miss so a silently skipped edit is impossible. It also reverts two
caption edits made earlier: the separately trained diagnostic checkpoint turns
out to produce predictions identical to the 2x2 ablation's 3-channel PhysPrior
cell on all 197 test samples, so the two rows are not different runs after all.

Run:  python -X utf8 scripts/patch_manuscript_deterministic.py
Then: python -X utf8 scripts/verify_manuscript_numbers.py
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = []   # (old, new, label)


def rep(old, new, label):
    R.append((old, new, label))


# ---------------------------------------------------------------- abstract
rep(r"reach $70.66\% \pm 3.74$~pp accuracy, versus $70.96\% \pm 2.64$~pp for the same with Co-Teaching; both are far above the $32.5$--$47.2\%$ reached",
    r"reach $70.36\% \pm 2.28$~pp accuracy, versus $74.52\% \pm 2.43$~pp when the same correction is combined with Co-Teaching; both are far above the $29.4$--$50.3\%$ reached",
    "abstract: five-seed + baseline range")

rep(r"corrected labels improve downstream accuracy by $21.3$~pp with two input channels and $31.5$~pp with three.",
    r"corrected labels improve downstream accuracy by $12.7$~pp with two input channels and $36.5$~pp with three.",
    "abstract: ablation deltas")

# ---------------------------------------------------------------- introduction
rep(r"achieves $70.66\% \pm 3.74$~pp accuracy across five random seeds, against $43.1\%$ for the same architecture trained on raw CLDMSK labels and $32.5$--$47.2\%$ for every noise-robust training method we tested",
    r"achieves $70.36\% \pm 2.28$~pp accuracy across five random seeds, against $41.1\%$ for the same architecture trained on raw CLDMSK labels and $29.4$--$50.3\%$ for every noise-robust training method we tested",
    "intro: contribution 4 headline")

rep(r"by $21.3$~pp with two channels and $31.5$~pp with three channels",
    r"by $12.7$~pp with two channels and $36.5$~pp with three channels",
    "intro: contribution 4 ablation deltas")

# ---------------------------------------------------------------- methods compared
rep(r"the same architecture reaches $75.6\%$ on PhysPrior-corrected labels against $43.1\%$ on raw labels, while loss correction, Co-Teaching, GCE and Mixup on raw labels reach only $32.5$--$47.2\%$. Second, the two Co-Teaching variants diverge sharply ($44.7\%$ versus $32.5\%$).",
    r"the same architecture reaches $71.6\%$ on PhysPrior-corrected labels against $41.1\%$ on raw labels, while loss correction, Co-Teaching, GCE and Mixup on raw labels reach only $29.4$--$50.3\%$. Second, Mixup collapses to the always-cloud solution ($29.4\%$ accuracy, $0.0\%$ clear recall).",
    "text before Table I")

rep(r"as a single-seed $2\times 2$ cell under three channels ($75.6\%$, Section~\ref{sec:abl}), and as the five-seed DL mean $70.66\%\pm 3.74$~pp",
    r"as a single-seed $2\times 2$ cell under three channels ($71.6\%$, Section~\ref{sec:abl}), and as the five-seed DL mean $70.36\%\pm 2.28$~pp",
    "text before Table I: PhysPrior appearances")

# ---------------------------------------------------------------- Table I
rep(r"Standard training & 197 & 43.1\%\,[36.5,\,50.3] & 59.7 & 50.9 & +0.257 \\",
    r"Standard training & 197 & 41.1\%\,[34.5,\,47.7] & 58.3 & 50.0 & +0.235 \\",
    "Table I: standard training")
rep(r"Loss Correction & 197 & 36.0\%\,[29.4,\,42.6] & 54.2 & 47.5 & +0.143 \\",
    r"Loss Correction & 197 & 50.3\%\,[43.1,\,57.4] & 64.2 & 53.8 & +0.314 \\",
    "Table I: loss correction")
rep(r"Co-Teaching A & 197 & 44.7\%\,[37.6,\,51.8] & 60.8 & 51.6 & +0.274 \\",
    r"Co-Teaching A & 197 & 39.6\%\,[33.0,\,46.7] & 56.2 & 48.5 & +0.172 \\",
    "Table I: co-teaching A")
rep(r"Co-Teaching B & 197 & 32.5\%\,[25.9,\,39.1] & 52.2 & 46.6 & +0.114 \\",
    r"Co-Teaching B & 197 & 38.6\%\,[32.0,\,45.2] & 55.5 & 48.1 & +0.158 \\",
    "Table I: co-teaching B")
rep(r"GCE ($q{=}0.7$) & 197 & 47.2\%\,[40.1,\,54.3] & 61.6 & 51.9 & +0.265 \\",
    r"GCE ($q{=}0.7$) & 197 & 39.6\%\,[33.0,\,46.7] & 56.2 & 48.5 & +0.172 \\",
    "Table I: GCE")
rep(r"Mixup & 197 & 37.6\%\,[31.0,\,44.2] & 54.8 & 47.7 & +0.143 \\",
    r"Mixup & 197 & 29.4\%\,[23.4,\,35.5] & 50.0 & 45.5 & +0.000 \\",
    "Table I: mixup")
rep(r"PhysPrior moderate + DL (3 channels) & 197 & \textbf{75.6\%}\,[69.5,\,81.7] & \textbf{73.7} & \textbf{62.5} & \textbf{+0.451} \\",
    r"PhysPrior moderate + DL (3 channels) & 197 & \textbf{71.6\%}\,[65.0,\,77.7] & \textbf{72.3} & \textbf{60.6} & \textbf{+0.411} \\",
    "Table I: PhysPrior + DL")
rep(r"PhysPrior + Co-Teaching (predef.\ ensemble) & 197 & 66.0\%\,[59.4,\,72.6] & 71.9 & 59.9 & +0.401 \\",
    r"PhysPrior + Co-Teaching (predef.\ ensemble) & 197 & 61.9\%\,[55.3,\,68.5] & 68.0 & 56.1 & +0.332 \\",
    "Table I: PhysPrior + Co-Teaching")

# ---------------------------------------------------------------- Table IV (station)
rep(r"Changsha (urban) & 29  & 48.3\% & 46.4\% & 100.0\% & 6.2\% \\",
    r"Changsha (urban) & 29  & 62.1\% & 60.0\% & 46.2\% & 75.0\% \\",
    "Table IV: Changsha row")
rep(r"Longmen (suburban) & 168 & 76.2\% & 53.7\% & 80.0\% & 74.8\% \\",
    r"Longmen (suburban) & 168 & 73.2\% & 50.0\% & 82.2\% & 69.9\% \\",
    "Table IV: Longmen row")
rep(r"Combined & 197 & 72.1\% & 51.6\% & 84.5\% & 66.9\% \\",
    r"Combined & 197 & 71.6\% & 51.2\% & 74.1\% & 70.5\% \\",
    "Table IV: combined row")

# ---------------------------------------------------------------- Table IV-b (season)
rep(r"Spring (Mar--May) & 73.2\% & [60.7, 83.9] & 56 \\",
    r"Spring (Mar--May) & 71.4\% & [58.9, 82.1] & 56 \\",
    "Table IV-b: spring")
rep(r"Summer (Jun--Aug) & 73.6\% & [62.5, 83.3] & 72 \\",
    r"Summer (Jun--Aug) & 69.4\% & [58.3, 80.6] & 72 \\",
    "Table IV-b: summer")
rep(r"Autumn (Sep--Nov) & 85.0\% & [72.5, 95.0] & 40 \\",
    r"Autumn (Sep--Nov) & 82.5\% & [70.0, 92.5] & 40 \\",
    "Table IV-b: autumn")
rep(r"Winter (Dec--Feb) & 48.3\% & [31.0, 65.5] & 29 \\",
    r"Winter (Dec--Feb) & 62.1\% & [44.8, 79.3] & 29 \\",
    "Table IV-b: winter")

# ---------------------------------------------------------------- Table V (2x2)
rep(r"Raw CLDMSK            & 47.7\%  & 44.2\% \\",
    r"Raw CLDMSK            & 49.7\%  & 35.0\% \\",
    "Table V: raw cells")
rep(r"PhysPrior moderate    & 69.0\%  & \textbf{75.6\%} \\",
    r"PhysPrior moderate    & 62.4\%  & \textbf{71.6\%} \\",
    "Table V: PhysPrior cells")
rep(r"\quad Under Raw CLDMSK  & \multicolumn{2}{r}{$-3.55$~pp} \\",
    r"\quad Under Raw CLDMSK  & \multicolumn{2}{r}{$-14.72$~pp} \\",
    "Table V: basemap effect under raw")
rep(r"\quad Under PhysPrior   & \multicolumn{2}{r}{$+6.60$~pp} \\",
    r"\quad Under PhysPrior   & \multicolumn{2}{r}{$+9.14$~pp} \\",
    "Table V: basemap effect under PhysPrior")
rep(r"\quad Under 2-ch        & \multicolumn{2}{r}{$+21.32$~pp} \\",
    r"\quad Under 2-ch        & \multicolumn{2}{r}{$+12.69$~pp} \\",
    "Table V: PhysPrior effect 2ch")
rep(r"\quad Under 3-ch        & \multicolumn{2}{r}{$+31.47$~pp} \\",
    r"\quad Under 3-ch        & \multicolumn{2}{r}{$+36.55$~pp} \\",
    "Table V: PhysPrior effect 3ch")

rep(r"PhysPrior label correction dominates both channel configurations ($+21.3$~pp at 2-channel and $+31.5$~pp at 3-channel); on this training set the choice of labels matters several times more than the choice of channels. Second, the basemap channel behaves differently under the two label regimes: it costs $3.6$~pp under raw labels but gains $6.6$~pp under corrected labels.",
    r"PhysPrior label correction improves downstream accuracy under both channel configurations ($+12.7$~pp at 2-channel and $+36.5$~pp at 3-channel); on this training set the choice of labels matters more than the choice of channels under both label regimes. Second, the basemap channel behaves differently under the two label regimes: it costs $14.7$~pp under raw labels but gains $9.1$~pp under corrected labels.",
    "Table V narrative")

# ---------------------------------------------------------------- Table VI (agreement + five seeds)
rep(r"Downstream DL correct against radar                & 142/197 (72.08\%) \\",
    r"Downstream DL correct against radar                & 141/197 (71.57\%) \\",
    "Table VI: DL correct")
rep(r"\emph{Both} correct against radar                   & 124/197 (62.94\%) \\",
    r"\emph{Both} correct against radar                   & 120/197 (60.91\%) \\",
    "Table VI: both correct")
rep(r"Direct PhysPrior only correct                     & 13/197 \\",
    r"Direct PhysPrior only correct                     & 17/197 \\",
    "Table VI: rule only")
rep(r"Downstream DL only correct                        & 18/197 \\",
    r"Downstream DL only correct                        & 21/197 \\",
    "Table VI: DL only")
rep(r"Both incorrect                                    & 42/197 \\",
    r"Both incorrect                                    & 39/197 \\",
    "Table VI: both incorrect")
rep(r"Sample-wise agreement (label $=$ model)           & 166/197 (84.26\%) \\",
    r"Sample-wise agreement (label $=$ model)           & 159/197 (80.71\%) \\",
    "Table VI: agreement")
rep(r"Cohen's $\kappa$                                   & $+0.684$ \\",
    r"Cohen's $\kappa$                                   & $+0.608$ \\",
    "Table VI: kappa")
rep(r"McNemar discordant counts $(b,c)$                 & $(19, 12)$ \\",
    r"McNemar discordant counts $(b,c)$                 & $(17, 21)$ \\",
    "Table VI: McNemar counts")
rep(r"Exact McNemar $p$-value (two-sided)               & $0.281$ \\",
    r"Exact McNemar $p$-value (two-sided)               & $0.627$ \\",
    "Table VI: McNemar p")
rep(r"Five-seed mean accuracy (PhysPrior$+$DL)             & $70.66\% \pm 3.74$~pp \\",
    r"Five-seed mean accuracy (PhysPrior$+$DL)             & $70.36\% \pm 2.28$~pp \\",
    "Table VI: five-seed accuracy")
rep(r"Five-seed mean balanced accuracy (PhysPrior$+$DL)         & $73.88\% \pm 1.68$~pp \\",
    r"Five-seed mean balanced accuracy (PhysPrior$+$DL)         & $72.16\% \pm 1.19$~pp \\",
    "Table VI: five-seed BA")
rep(r"Five-seed mean F1 (PhysPrior$+$DL)                       & $62.25\% \pm 2.03$~pp \\",
    r"Five-seed mean F1 (PhysPrior$+$DL)                       & $60.26\% \pm 1.22$~pp \\",
    "Table VI: five-seed F1")
rep(r"Five-seed mean recall (PhysPrior$+$DL)                  & $81.72\% \pm 5.73$~pp \\",
    r"Five-seed mean recall (PhysPrior$+$DL)                  & $76.55\% \pm 7.76$~pp \\",
    "Table VI: five-seed recall")
rep(r"Five-seed mean specificity (PhysPrior$+$DL)             & $66.04\% \pm 7.33$~pp \\",
    r"Five-seed mean specificity (PhysPrior$+$DL)             & $67.77\% \pm 6.25$~pp \\",
    "Table VI: five-seed specificity")

# ---------------------------------------------------------------- discussion text
rep(r"The downstream model agrees with the direct rule on $84.26\%$ of the $197$ pixels ($\kappa = +0.684$, McNemar $b = 19$, $c = 12$, $p = 0.281$; $124$ both correct, $13$ rule only, $18$ DL only). On the $31$ discordant pixels the model is correct $18$ times and the rule $13$. On the Changsha subset it predicts cloud for almost every patch ($100\%$ recall, $6.2\%$ specificity).",
    r"The downstream model agrees with the direct rule on $80.71\%$ of the $197$ pixels ($\kappa = +0.608$, McNemar $b = 17$, $c = 21$, $p = 0.627$; $120$ both correct, $17$ rule only, $21$ DL only). On the $38$ discordant pixels the model is correct $21$ times and the rule $17$. On the Changsha subset it under-predicts cloud ($46.2\%$ recall, $75.0\%$ specificity).",
    "agreement narrative")

rep(r"The five-seed PhysPrior$+$DL mean is $70.66\% \pm 3.74$~pp accuracy (range $66.50$--$77.16$\%), exceeding the raw-CLDMSK baseline by $26.5$~pp. It is indistinguishable from the always-clear $70.6\%$ point estimate in accuracy alone, but the always-clear classifier attains that figure with $0\%$ cloud recall against PhysPrior's $81.7\%$. Repeating the protocol with identical seeds moved individual seeds by up to $4$~pp, so five seeds understate the run-to-run spread and the interval is a lower bound. A paired-seed comparison remains the natural next step.",
    r"The five-seed PhysPrior$+$DL mean is $70.36\% \pm 2.28$~pp accuracy (range $66.50$--$73.10$\%), exceeding the raw-CLDMSK baseline by $26.2$~pp. It is indistinguishable from the always-clear $70.6\%$ point estimate in accuracy alone, but the always-clear classifier attains that figure with $0\%$ cloud recall against PhysPrior's $76.6\%$. Combining the same corrected labels with Co-Teaching raises the five-seed mean to $74.52\% \pm 2.43$~pp, $4.16$~pp above the plain network on the same five seeds. A paired-seed PhysPrior-versus-raw comparison remains the natural next step.",
    "five-seed narrative")

rep(r"Seasonal accuracy ranges from $48.3\%$ winter to $85.0\%$ autumn (Table~\ref{tab:season})",
    r"Seasonal accuracy ranges from $62.1\%$ winter to $82.5\%$ autumn (Table~\ref{tab:season})",
    "seasonal narrative")

# ---------------------------------------------------------------- limitations
rep(r"(ii)~Two sites and $n{=}197$ test patches yield wide confidence intervals; multi-year, multi-climate validation is required.",
    r"(ii)~Two sites and $n{=}197$ test patches yield wide confidence intervals, and the Longmen radar archive covers 2020 only, so multi-year validation requires an external reference.",
    "limitation (ii): Longmen radar coverage")

# ---------------------------------------------------------------- conclusion
rep(r"models trained on the corrected labels reach $70.66\% \pm 3.74$~pp accuracy across five seeds, against $43.1\%$ on raw labels and $32.5$--$47.2\%$ for four noise-robust training methods. PhysPrior is a label-bias correction framework, not an operational CLDMSK replacement: the matched M15 threshold ($76.6\%$) attains higher raw accuracy, and the incremental value of Co-Teaching on top of the corrected labels is not resolved by five seeds ($70.96\% \pm 2.64$~pp).",
    r"models trained on the corrected labels reach $70.36\% \pm 2.28$~pp accuracy across five seeds, against $41.1\%$ on raw labels and $29.4$--$50.3\%$ for four noise-robust training methods. PhysPrior is a label-bias correction framework, not an operational CLDMSK replacement: the matched M15 threshold ($76.6\%$) attains higher raw accuracy, and combining the corrected labels with Co-Teaching adds a further $4.16$~pp ($74.52\% \pm 2.43$~pp).",
    "conclusion")

# ---------------------------------------------------------------- caption reverts
rep(r"\caption{Per-station breakdown of a separately trained single-checkpoint PhysPrior$+$DL moderate model used",
    r"\caption{Per-station breakdown of the saved single-checkpoint PhysPrior$+$DL moderate model used",
    "revert Table IV caption")
rep(r"\caption{Per-season breakdown (calendar-accurate months) of the separately trained single-checkpoint PhysPrior$+$DL moderate model;",
    r"\caption{Per-season breakdown (calendar-accurate months) of the saved single-checkpoint PhysPrior$+$DL moderate model;",
    "revert Table IV-b caption")


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
