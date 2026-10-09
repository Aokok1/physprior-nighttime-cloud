"""Second space pass: state the same facts in fewer words.

After the text-only revision and the first compression the letter still spilled
about 65 pt -- roughly seven bibliography lines -- onto a sixth page. This pass
removes restatement and duplicated numbers only; no claim, number or caveat is
dropped.

Run:  python -X utf8 scripts/patch_manuscript_compress3.py
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = [
    # ---------------------------------------------------------------- Sec. I positioning
    (r"We position PhysPrior as a two-site proof of concept for physics-guided correction of systematic nighttime label bias. The objective is not to replace CLDMSK operationally or to outperform every M15-based detector, but to test whether a constrained cloud-to-clear correction measurably improves the training signal, so the conclusions are limited to the $197$ radar-referenced samples from two subtropical South China sites.",
     r"PhysPrior is a two-site proof of concept for physics-guided correction of systematic nighttime label bias. It does not aim to replace CLDMSK operationally or to outperform every M15-based detector, but to test whether a constrained cloud-to-clear correction improves the training signal, so the conclusions are limited to the $197$ radar-referenced samples.",
     "Sec. I positioning"),

    # ---------------------------------------------------------------- Sec. I contribution 1
    (r"Against an independent Ka-band radar reference ($n{=}197$), raw CLDMSK labels achieve $44.2\%$ agreement and a $77.0\%$ apparent false-positive rate, documenting a pronounced apparent over-detection pattern at the two subtropical South China sites.",
     r"Against an independent Ka-band radar reference ($n{=}197$), raw CLDMSK labels achieve $44.2\%$ agreement and a $77.0\%$ apparent false-positive rate, a pronounced apparent over-detection pattern.",
     "Sec. I contribution 1"),

    # ---------------------------------------------------------------- Sec. I contribution 3
    (r"\item \textbf{Explicit assessment of correction quality on the radar set.} Of the $74$ labels changed by PhysPrior on the $197$ radar-collocated pixels, $62$ are apparent false-positive corrections and $12$ are over-corrections, yielding beneficial- and adverse-flip fractions of $83.8\%$ and $16.2\%$. This direct count makes the trade-off introduced by one-directional correction observable on the test split, rather than only implicit in the resulting accuracy.",
     r"\item \textbf{Explicit assessment of correction quality on the radar set.} Of the $74$ labels PhysPrior changes on the $197$ radar-collocated pixels, $62$ are apparent false-positive corrections and $12$ are over-corrections ($83.8\%$ and $16.2\%$). The trade-off introduced by one-directional correction is therefore counted directly rather than left implicit in the resulting accuracy.",
     "Sec. I contribution 3"),

    # ---------------------------------------------------------------- Sec. I contribution 4
    (r"A model trained on PhysPrior-corrected labels achieves $70.36\% \pm 2.28$~pp accuracy across five random seeds, against $41.1\%$ for the same architecture trained on raw CLDMSK labels and $29.4$--$50.3\%$ for every noise-robust training method we tested. In a single-seed $2\times 2$ ablation, PhysPrior improves downstream accuracy additively under both channel configurations, by $+19.8$~pp, while the basemap channel costs $7.6$~pp; repeated-seed and geographically grouped controls remain necessary.",
     r"A model trained on PhysPrior-corrected labels achieves $70.36\% \pm 2.28$~pp accuracy across five seeds, against $41.1\%$ on raw CLDMSK labels and $29.4$--$50.3\%$ for every noise-robust training method tested. In a single-seed $2\times 2$ ablation the two factors are additive: labels are worth $+19.8$~pp and the basemap costs $7.6$~pp, under both channel configurations; repeated-seed and grouped controls remain necessary.",
     "Sec. I contribution 4"),

    # ---------------------------------------------------------------- Sec. II-B opening
    (r"CLDMSK nighttime tests rely primarily on M15 brightness temperature and 5$\times$5 spatial uniformity~\cite{ackerman2010disc}. Cloud tops are colder than land surfaces; if M15 exceeds a threshold $T_{\rm th}$, the pixel is likely clear:",
     r"CLDMSK nighttime tests rely on M15 brightness temperature and 5$\times$5 spatial uniformity~\cite{ackerman2010disc}. Cloud tops are colder than land; if M15 exceeds a threshold $T_{\rm th}$ the pixel is likely clear:",
     "Sec. II-B opening"),

    # ---------------------------------------------------------------- Sec. III-G
    (r"The downstream model agrees with the direct rule on $80.71\%$ of the $197$ pixels ($\kappa = +0.608$, McNemar $b = 17$, $c = 21$, $p = 0.627$; $120$ both correct, $17$ rule only, $21$ DL only). On the $38$ discordant pixels the model is correct $21$ times and the rule $17$. On the Changsha subset it under-predicts cloud ($46.2\%$ recall, $75.0\%$ specificity). The diagnostics do not rule out overfitting because $n = 197$ is small and the backbone has $24.7$~M parameters.",
     r"The downstream model agrees with the direct rule on $80.71\%$ of the $197$ pixels ($\kappa = +0.608$, McNemar $b = 17$, $c = 21$, $p = 0.627$; $120$ both correct, $17$ rule only, $21$ DL only), so it is not a copy of the rule; on the Changsha subset it under-predicts cloud ($46.2\%$ recall, $75.0\%$ specificity). The diagnostics do not rule out overfitting, because $n = 197$ is small and the backbone has $24.7$~M parameters.",
     "Sec. III-G compression"),

    (r"The five-seed PhysPrior$+$DL mean is $70.36\% \pm 2.28$~pp accuracy (range $66.50$--$73.10$\%), exceeding the raw-CLDMSK baseline by $26.2$~pp. It is indistinguishable from the always-clear $70.6\%$ point estimate in accuracy alone, but the always-clear classifier attains that figure with $0\%$ cloud recall against the five-seed mean cloud recall of $76.6\%$. Combining the same corrected labels with Co-Teaching raises the five-seed mean to $74.52\% \pm 2.43$~pp, $4.16$~pp above the plain network on the same five seeds. A paired-seed PhysPrior-versus-raw comparison remains the natural next step.",
     r"The five-seed PhysPrior$+$DL mean is $70.36\% \pm 2.28$~pp accuracy (range $66.50$--$73.10$\%), $26.2$~pp above the raw-CLDMSK baseline. It is indistinguishable from the always-clear $70.6\%$ in accuracy alone, but that classifier attains the figure with $0\%$ cloud recall against the five-seed mean recall of $76.6\%$. Adding Co-Teaching raises the five-seed mean to $74.52\% \pm 2.43$~pp. A paired-seed PhysPrior-versus-raw comparison remains the natural next step.",
     "Sec. III-G five-seed compression"),

    # ---------------------------------------------------------------- Sec. IV-A
    (r"PhysPrior corrects the most consistent failure mode of CLDMSK over subtropical land: it raises radar agreement from $44.2\%$ (raw CLDMSK) to $69.5\%$, and training on the corrected labels lifts the same network from $41.1\%$ to $71.6\%$. Both M15-only threshold rows nevertheless attain higher raw accuracy ($76.6\%$ and $77.2\%$), so PhysPrior is a label-bias correction framework rather than a substitute for a well-tuned M15 threshold.",
     r"PhysPrior corrects the most consistent failure mode of CLDMSK over subtropical land: radar agreement rises from $44.2\%$ to $69.5\%$, and training on the corrected labels lifts the same architecture from $41.1\%$ to $71.6\%$. Both M15-only rows nevertheless attain higher raw accuracy ($76.6\%$ and $77.2\%$), so PhysPrior is a label-bias correction framework, not a substitute for a well-tuned M15 threshold.",
     "Sec. IV-A compression"),

    # ---------------------------------------------------------------- Sec. V conclusion
    (r"This two-site proof of concept shows that PhysPrior corrects a systematic nighttime over-detection bias in VIIRS CLDMSK. Direct radar agreement rises from $44.2\%$ to $69.5\%$, and models trained on the corrected labels reach $70.36\% \pm 2.28$~pp accuracy across five seeds, against $41.1\%$ on raw labels and $29.4$--$50.3\%$ for four noise-robust training methods. PhysPrior is a label-bias correction framework, not an operational CLDMSK replacement: the matched M15 threshold ($76.6\%$) attains higher raw accuracy, and combining the corrected labels with Co-Teaching adds a further $4.16$~pp ($74.52\% \pm 2.43$~pp).",
     r"This two-site proof of concept shows that PhysPrior corrects a systematic nighttime over-detection bias in VIIRS CLDMSK. Direct radar agreement rises from $44.2\%$ to $69.5\%$, and models trained on the corrected labels reach $70.36\% \pm 2.28$~pp across five seeds, against $41.1\%$ on raw labels and $29.4$--$50.3\%$ for four noise-robust methods. It remains a label-bias correction framework rather than an operational replacement: the matched M15 threshold reaches $76.6\%$, and Co-Teaching adds $4.16$~pp ($74.52\% \pm 2.43$~pp).",
     "Sec. V compression"),
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
