"""Reclaim the space the reviewer fixes cost, without dropping any of them.

The R2-1 / R2-5 / R2-8 additions plus two references pushed the letter to six
pages. GRSL allows five including references, so the same material is stated
more tightly here: every compression below keeps the fact and drops restatement.

Run:  python -X utf8 scripts/patch_manuscript_compress.py
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = [
    # ---- Sec. II-A: two label spaces, stated once ------------------------
    (r"Let $\mathbf{x}\in\mathbb{R}^{C\times H\times W}$ denote a VIIRS patch and $y\in\{\mathrm{clear},\mathrm{cloud}\}$ the binary radar-collocated target. Two label spaces are used in this paper. The \emph{derived CLDMSK label space} $\tilde{y}\in\{0,1,2,3\}$ used here encodes confidence in the reverse bit order of the product's own \texttt{Integer\_Cloud\_Mask}, so that $\tilde{y}=3$ is the confident cloud class and $\tilde{y}=0$ the confident clear class; the division into cloud and clear is identical either way, with $\{\tilde{y}\ge 2\}$ corresponding to the product's $\{$cloudy, probably cloudy$\}$. All segmentation networks use a four-class output head; during training the per-pixel cross-entropy is computed against the network-specific target $\hat{y}\in\{0,1,2,3\}$, which is either $\hat{y}=\tilde{y}$ (raw-CLDMSK training, used for the ``Standard training'' row of Table~\ref{tab:methods}) or $\hat{y}=\tilde{y}^{\rm PP}$ (PhysPrior-corrected training, used for every PhysPrior-prefixed deep-learning row of Table~\ref{tab:methods}). The PhysPrior correction reassigns both cloud classes $\{2,3\}$ to the single clear class $0$, which is the lowest-confidence clear class and the assignment made by the reference implementation. For binary evaluation against the radar reference, the predicted class $\ge 2$ is mapped to cloud, so all reported numbers---direct CLDMSK, direct PhysPrior, all DL predictions and the M15 thresholds---are on the same binary cloud/clear task.",
     r"Let $\mathbf{x}\in\mathbb{R}^{C\times H\times W}$ denote a VIIRS patch and $y\in\{\mathrm{clear},\mathrm{cloud}\}$ the binary radar-collocated target. The \emph{derived CLDMSK label space} $\tilde{y}\in\{0,1,2,3\}$ encodes confidence in the reverse bit order of the product's own \texttt{Integer\_Cloud\_Mask}, so $\tilde{y}=3$ is the confident cloud class; the cloud/clear division is identical either way, with $\{\tilde{y}\ge 2\}$ matching the product's $\{$cloudy, probably cloudy$\}$. All segmentation networks use a four-class head trained against a network-specific target $\hat{y}\in\{0,1,2,3\}$: $\hat{y}=\tilde{y}$ for the raw-CLDMSK ``Standard training'' row of Table~\ref{tab:methods}, or $\hat{y}=\tilde{y}^{\rm PP}$ for every PhysPrior-prefixed deep-learning row. The correction reassigns both cloud classes $\{2,3\}$ to class $0$. For binary evaluation the predicted class $\ge 2$ is mapped to cloud, so direct CLDMSK, direct PhysPrior, all DL predictions and the M15 thresholds are scored on one binary task.",
     "Sec. II-A compression"),

    # ---- Sec. II-D: radar verification, stated once ----------------------
    (r"The radar verification procedure is as follows. Each VIIRS test patch is colocated with the nearest Ka-band radar pixel within $\pm 10$~minutes, and the radar label is the binary call from any echo exceeding $-33$~dBZ within the lowest 10~km. Direct methods (raw CLDMSK, M15 thresholds, the direct PhysPrior rule) are then compared at the patch center pixel against this single binary radar label. For the trained segmentation models the test-side prediction is instead aggregated over the $3\times 3$ patch neighborhood around the radar pixel using a logical-OR rule (any predicted class index $\ge 2$ inside the window sets the patch-level call to cloud); this larger spatial sampling compensates for the $\pm 10$-min temporal and the residual geolocation mismatch within the $128 \times 128$ VIIRS patch; the $3\times 3$ aggregation is a geolocation-tolerance window, not a vertical-echo analog of the radar label. The two protocols---center-pixel for direct methods, $3\times 3$ OR for DL models---are reported separately in Table~\ref{tab:methods} via the direct-method and deep-learning blocks; aggregate cross-block comparisons should be read with this protocol split in mind. Ka-band radar provides an active-sensor reference physically distinct from VIIRS thermal-infrared radiance, supporting the evaluation independence; using ground-based radar to validate satellite cloud products is a standard practice~\cite{huohan2015modis}. Known limitations include reduced sensitivity to very thin cirrus ($Z<-35$~dBZ) and attenuation in heavy precipitation; we address these in Section~\ref{sec:limit}.",
     r"Each VIIRS test patch is colocated with the nearest Ka-band radar pixel within $\pm 10$~minutes, and the radar label is the binary call from any echo exceeding $-33$~dBZ within the lowest 10~km. Direct methods (raw CLDMSK, M15 thresholds, the direct PhysPrior rule) are compared at the patch center pixel against this label. Trained segmentation models are instead aggregated over the $3\times 3$ neighbourhood around the radar pixel by a logical OR (any predicted class index $\ge 2$ sets cloud), a geolocation-tolerance window covering the $\pm 10$-min temporal and residual spatial mismatch rather than a vertical-echo analog of the radar label. The two protocols are reported separately in Table~\ref{tab:methods}; cross-block comparisons should be read with that split in mind. Ka-band radar is an active-sensor reference physically distinct from VIIRS thermal-infrared radiance, and validating satellite cloud products against ground-based radar is standard practice~\cite{huohan2015modis}. Reduced sensitivity to very thin cirrus ($Z<-35$~dBZ) and attenuation in heavy precipitation are addressed in Section~\ref{sec:limit}.",
     "Sec. II-D compression"),

    # ---- Sec. III-B: methods compared ------------------------------------
    (r"Table~\ref{tab:methods} lists all methods evaluated on the radar-reference test set; every confusion-matrix entry and bootstrap $95\%$ interval is computed from the per-sample prediction records of the constituent runs. The set includes two trivial baselines, two Pure-M15 threshold rows ($T{=}264$~K matched to PhysPrior and $T{=}266$~K as post-hoc oracle), three direct PhysPrior ablation presets, and eight trained deep-learning configurations. Two observations organise the deep-learning block. First, the choice of labels dominates the choice of training method:",
     r"Table~\ref{tab:methods} lists every method evaluated on the radar-reference test set, each computed from the per-sample prediction records of its constituent runs: two trivial baselines, two Pure-M15 threshold rows ($T{=}264$~K matched to PhysPrior and $T{=}266$~K as post-hoc oracle), three direct PhysPrior presets, and eight trained deep-learning configurations. Two observations organise the deep-learning block. First, the choice of labels dominates the choice of training method:",
     "Sec. III-B compression"),

    # ---- Sec. I: proof-of-concept positioning ----------------------------
    (r"We position PhysPrior as a two-site proof of concept for physics-guided correction of systematic nighttime label bias. The objective is not to replace CLDMSK operationally or to outperform every M15-based detector, but to test whether a constrained cloud-to-clear correction measurably improves the training signal. The conclusions are therefore limited to the $197$ radar-referenced samples from two subtropical South China sites.",
     r"We position PhysPrior as a two-site proof of concept for physics-guided correction of systematic nighttime label bias. The objective is not to replace CLDMSK operationally or to outperform every M15-based detector, but to test whether a constrained cloud-to-clear correction measurably improves the training signal, so the conclusions are limited to the $197$ radar-referenced samples from two subtropical South China sites.",
     "Sec. I compression"),

    # ---- Sec. IV-A: what PhysPrior does and does not demonstrate ---------
    (r"PhysPrior corrects the most consistent failure mode of CLDMSK over subtropical land. Applied to the test set it raises radar agreement from $44.2\%$ (raw CLDMSK) to $69.5\%$, and training on the corrected labels lifts the same network from $41.1\%$ to $71.6\%$. Both M15-only threshold rows nevertheless attain higher raw accuracy ($76.6\%$ and $77.2\%$), so PhysPrior is a label-bias correction framework rather than a substitute for a well-tuned M15 threshold.",
     r"PhysPrior corrects the most consistent failure mode of CLDMSK over subtropical land: it raises radar agreement from $44.2\%$ (raw CLDMSK) to $69.5\%$, and training on the corrected labels lifts the same network from $41.1\%$ to $71.6\%$. Both M15-only threshold rows nevertheless attain higher raw accuracy ($76.6\%$ and $77.2\%$), so PhysPrior is a label-bias correction framework rather than a substitute for a well-tuned M15 threshold.",
     "Sec. IV-A compression"),

    # ---- Sec. III-C: M15 sensitivity benchmark ---------------------------
    (r"The $T{=}266$~K M15 row achieves $77.2\%$ accuracy and is the optimum of a sensitivity scan over $\{260, 262, 264, 266\}$~K (full sweep in the supplementary material) and is therefore labelled a \emph{post-hoc test-set oracle} benchmark, not an operational baseline. The primary M15-only reference is the matched $T{=}264$~K row, which reaches $76.6\%$ accuracy, $0.5$~pp below the oracle. Both thresholds are fixed on the radar-reference test set.",
     r"The $T{=}266$~K M15 row achieves $77.2\%$ accuracy, the optimum of a sensitivity scan over $\{260, 262, 264, 266\}$~K (full sweep in the supplementary material), and is therefore labelled a \emph{post-hoc test-set oracle} benchmark rather than an operational baseline. The primary M15-only reference is the matched $T{=}264$~K row at $76.6\%$, $0.5$~pp below the oracle. Both thresholds are fixed on the radar-reference test set.",
     "Sec. III-C compression"),

    # ---- Sec. IV-B: seasonal confounding ---------------------------------
    (r"The winter value is consistent with reduced cloud--surface thermal contrast but could also reflect threshold mismatch, cloud-type differences, or sampling variability; the seasonal results are exploratory rather than climatologically representative.",
     r"The winter value is consistent with reduced cloud--surface thermal contrast but could also reflect threshold mismatch, cloud-type differences, or sampling variability, so these results are exploratory rather than climatologically representative.",
     "Sec. IV-B compression"),
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
