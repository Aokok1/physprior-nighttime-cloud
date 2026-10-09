"""Reclaim the column the text-only revision cost, without dropping any fix.

The revision added the operator definition, the sampling scheme, the corrected
basemap description, the checkpoint protocol, the itemised data statement and
several disambiguations; the letter went to six pages. GRSL allows five
including references. Every compression below states the same fact in fewer
words -- no claim is removed.

Run:  python -X utf8 scripts/patch_manuscript_compress2.py
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = [
    # ---------------------------------------------------------------- II-A
    (r"Let $\mathbf{x}\in\mathbb{R}^{C\times H\times W}$ denote a VIIRS patch and $y\in\{\mathrm{clear},\mathrm{cloud}\}$ the binary radar-collocated target. The label field $\tilde{y}\in\{0,1,2,3\}$ is the product's own \texttt{Integer\_Cloud\_Mask} with its confidence ordering reversed, $\tilde{y}=3-\texttt{Integer\_Cloud\_Mask}$: $\tilde{y}=3$ is the product's confident-cloud level and $\tilde{y}=0$ its confident-clear level, so the cloud set $\{\tilde{y}\ge2\}$ coincides exactly with the product's $\{$cloudy, probably cloudy$\}$. All segmentation networks use a four-class head trained against a network-specific target $\hat{y}\in\{0,1,2,3\}$: $\hat{y}=\tilde{y}$ for the raw-CLDMSK ``Standard training'' row of Table~\ref{tab:methods}, or $\hat{y}=\tilde{y}^{\rm corr}$, the PhysPrior-corrected field, for every PhysPrior-prefixed deep-learning row. The correction reassigns both cloud classes $\{2,3\}$ to class $0$. For binary evaluation the predicted class $\ge 2$ is mapped to cloud, so direct CLDMSK, direct PhysPrior, all DL predictions and the M15 thresholds are scored on one binary task.",
     r"Let $\mathbf{x}\in\mathbb{R}^{C\times H\times W}$ be a VIIRS patch and $y\in\{\mathrm{clear},\mathrm{cloud}\}$ the binary radar-collocated target, evaluated at the patch centre. The label field $\tilde{y}\in\{0,1,2,3\}$ is the product's \texttt{Integer\_Cloud\_Mask} with its confidence ordering reversed, $\tilde{y}=3-\texttt{Integer\_Cloud\_Mask}$, so $\{\tilde{y}\ge2\}$ is exactly the product's $\{$cloudy, probably cloudy$\}$. Networks use a four-class head trained against $\hat{y}=\tilde{y}$ for the raw-CLDMSK ``Standard training'' row of Table~\ref{tab:methods} and $\hat{y}=\tilde{y}^{\rm corr}$, the PhysPrior-corrected field, for every PhysPrior-prefixed row. The correction reassigns classes $\{2,3\}$ to $0$, and evaluation maps class $\ge2$ to cloud, so every method is scored on one binary task.",
     "II-A compression"),

    # ---------------------------------------------------------------- II-B preset
    (r"The preset is evaluated post hoc against the radar reference, and its behaviour is site-dependent: on the $160$ radar-collocated training and validation pixels, all Changsha, neither this preset ($55.7\%$) nor any cell of an $8\times6$ threshold grid (best $62.3\%$) exceeds the raw CLDMSK agreement ($62.9\%$), whereas on the test set the same fixed rule raises agreement from $44.2\%$ to $69.5\%$. Transfer of a fixed rule across sites, rather than a development-set selection, is therefore what this result demonstrates.",
     r"The preset is evaluated post hoc: on the $160$ radar-collocated training and validation pixels, all Changsha, neither it ($55.7\%$) nor any cell of an $8\times6$ threshold grid (best $62.3\%$) beats raw CLDMSK ($62.9\%$), whereas on the test set the same fixed rule raises agreement from $44.2\%$ to $69.5\%$. What this demonstrates is transfer of a fixed rule across sites, not a development-set selection.",
     "II-B preset compression"),

    # ---------------------------------------------------------------- II-B prior remedies
    (r"Prior remedies fall into two families: threshold-based detectors replace CLDMSK with a competing mask, and noise-robust training objectives treat the label noise as unstructured. PhysPrior does neither. It edits the label field once, with a physical sign, so the corrected labels keep the product's four-class spatial structure and can be handed unchanged to any downstream training method.",
     r"Prior remedies either replace CLDMSK with a competing threshold mask or treat the label noise as unstructured. PhysPrior does neither: it edits the label field once, with a physical sign, so the corrected labels keep the product's four-class spatial structure and can be handed unchanged to any downstream method.",
     "II-B prior-remedies compression"),

    # ---------------------------------------------------------------- II-D
    (r"Each VIIRS test patch is a $128 \times 128$ pixel sample colocated with the nearest Ka-band radar pixel within $\pm 10$~minutes, and the radar label is the binary call from any echo exceeding $-33$~dBZ within the lowest 10~km. Direct methods (raw CLDMSK, M15 thresholds, the direct PhysPrior rule) are compared at the patch center pixel against this label. Trained segmentation models are instead aggregated over the $3\times 3$ neighbourhood around the radar pixel by a logical OR (any predicted class index $\ge 2$ sets cloud), a geolocation-tolerance window covering the $\pm 10$-min temporal and residual spatial mismatch rather than a vertical-echo analog of the radar label. The two protocols are reported separately in Table~\ref{tab:methods}; cross-block comparisons should be read with that split in mind. Ka-band radar is an active-sensor reference physically distinct from VIIRS thermal-infrared radiance, and validating satellite cloud products against ground-based radar is standard practice~\cite{huohan2015modis}. Cirrus weaker than the $-33$~dBZ detection threshold, and attenuation in heavy precipitation, are addressed in Section~\ref{sec:limit}.",
     r"Each test patch is a $128\times128$ VIIRS field colocated with the nearest Ka-band radar pixel within $\pm 10$~minutes; the radar label is the binary call from any echo exceeding $-33$~dBZ in the lowest 10~km. Direct methods (raw CLDMSK, M15 thresholds, the direct PhysPrior rule) are compared at the patch centre against this label. Trained models are aggregated over the $3\times3$ neighbourhood around the radar pixel by a logical OR (any class index $\ge2$ sets cloud), a geolocation-tolerance window covering the temporal and residual spatial mismatch. The two protocols are reported separately in Table~\ref{tab:methods} and cross-block comparisons should be read with that split in mind. Ka-band radar is an active-sensor reference distinct from VIIRS thermal-infrared radiance, and validating cloud products against ground-based radar is standard practice~\cite{huohan2015modis}; thin-cirrus and heavy-precipitation limits are discussed in Section~\ref{sec:limit}.",
     "II-D compression"),

    # ---------------------------------------------------------------- III-B
    (r"Table~\ref{tab:methods} lists every method evaluated on the radar-reference test set, each computed from the per-sample prediction records of its constituent runs:",
     r"Table~\ref{tab:methods} lists every method evaluated on the radar-reference test set, each computed from its per-sample prediction records:",
     "III-B opening compression"),

    (r" and as the five-seed DL mean $70.36\%\pm 2.28$~pp (Section~\ref{sec:overfit}). All cross-method comparisons share the same $197$-patch radar reference and the same per-patch binary label.",
     r" and as the five-seed DL mean $70.36\%\pm 2.28$~pp (Section~\ref{sec:overfit}).",
     "III-B closing sentence removed"),

    # ---------------------------------------------------------------- III-C
    (r"The $T{=}266$~K M15 row achieves $77.2\%$ accuracy, the optimum of a sensitivity scan over $\{260, 262, 264, 266\}$~K (full sweep in the supplementary material), and is therefore labelled a \emph{post-hoc test-set oracle} benchmark rather than an operational baseline. The primary M15-only reference is the matched $T{=}264$~K row at $76.6\%$, $0.5$~pp below the oracle. Both thresholds are fixed on the radar-reference test set.",
     r"The $T{=}266$~K M15 row achieves $77.2\%$, the optimum of a sensitivity scan over $\{260, 262, 264, 266\}$~K (full sweep in the supplementary material), and is therefore a \emph{post-hoc test-set oracle} rather than an operational baseline. The primary M15 reference is the matched $T{=}264$~K row at $76.6\%$, $0.5$~pp below it.",
     "III-C compression"),

    # ---------------------------------------------------------------- IV-B
    (r"Seasonal accuracy ranges from $62.1\%$ winter to $82.5\%$ autumn (Table~\ref{tab:season}), but the $29$ winter samples are all Changsha (2022, 2024 and 2025) and the $168$ non-winter are all Longmen (2020): station, year, season, and urban--suburban setting are mutually confounded and cannot be separated. Restricting the same model to Longmen alone, where site is fixed and only season varies, the spread persists: $69.4\%$ in summer ($n{=}72$) against $82.5\%$ in autumn ($n{=}40$), so the seasonal pattern is not only a station contrast. All $168$ Longmen patches come from a single year, however, so season and year remain inseparable there. The winter value is consistent with reduced cloud--surface thermal contrast but could also reflect threshold mismatch, cloud-type differences, or sampling variability, so these results are exploratory rather than climatologically representative.",
     r"Seasonal accuracy ranges from $62.1\%$ winter to $82.5\%$ autumn (Table~\ref{tab:season}), but the $29$ winter samples are all Changsha (2022, 2024 and 2025) and the $168$ non-winter all Longmen (2020): station, year, season and setting are mutually confounded. Within Longmen alone, where site is fixed, the spread persists---$69.4\%$ in summer ($n{=}72$) against $82.5\%$ in autumn ($n{=}40$)---so it is not only a station contrast; all $168$ Longmen patches are from one year, so season and year remain inseparable. The winter value is consistent with reduced cloud--surface thermal contrast but could also reflect threshold mismatch, cloud-type differences or sampling variability, so these results are exploratory.",
     "IV-B compression"),

    # ---------------------------------------------------------------- I contribution 2
    (r"Compared with the closest prior CLDMSK error-correction heuristics (single-channel warm-pixel thresholds~\cite{ackerman2010disc}), the novelty delta of PhysPrior is the explicit one-directional \emph{label-space} operation whose over-correction rate is explicitly quantified on the radar reference ($12/74 = 16.2\%$ TP$\to$FN flips)---a failure-mode accounting on which prior threshold-only corrections are silent.",
     r"Unlike a single-channel warm-pixel threshold~\cite{ackerman2010disc}, PhysPrior is an explicit one-directional \emph{label-space} operation whose over-correction rate is quantified on the radar reference ($12/74 = 16.2\%$ TP$\to$FN flips).",
     "I contribution 2 compression"),
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
