"""Text-only revision: close the comprehension and provenance gaps in five pages.

Implements the items from SIMULATED_REVIEW.md that need no new experiment. The
numbers affected by the basemap rebuild are handled separately once that run
finishes; everything here is number-independent.

  A1  sigma_5 operator undefined            -> defined, with a pointer to S1
  A2  the label-space sentence contradicts itself
                                            -> replaced with the verified mapping
                                               y = 3 - Integer_Cloud_Mask
  A3  yhat^PP undefined                     -> renamed
  A4  basemap called "static" and external  -> described as what it is
  A5  4x4 matrix with binary columns        -> 4x2 embedded in 4x4
  A6  -35 vs -33 dBZ looks inconsistent     -> phrased as below-threshold cirrus
  A7  checkpoint-selection rule unstated    -> stated
  A9  the +/- in Table VI undefined         -> defined
  A10 two different quantities both 76.6%   -> disambiguated
  A16 "binary labels" vs a four-class mask  -> reconciled
  A17 duplicated Table I row                -> removed
  A18 Co-Teaching A/B undefined             -> defined
  C16 patch sampling unexplained            -> stated (two fixed fields)
  B1  no platform / product                 -> stated
  B4  no split rule in the supplement chain -> data availability itemised
  Sec. 5 reference defects                  -> [2] corrected, [10] authors added

Run:  python -X utf8 scripts/patch_manuscript_textonly.py
Then: compile and check the page count.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = [
    # ---------------------------------------------------------------- A2, A3
    (r"The \emph{derived CLDMSK label space} $\tilde{y}\in\{0,1,2,3\}$ encodes confidence in the reverse bit order of the product's own \texttt{Integer\_Cloud\_Mask}, so $\tilde{y}=3$ is the confident cloud class; the cloud/clear division is identical either way, with $\{\tilde{y}\ge 2\}$ matching the product's $\{$cloudy, probably cloudy$\}$.",
     r"The label field $\tilde{y}\in\{0,1,2,3\}$ is the product's own \texttt{Integer\_Cloud\_Mask} with its confidence ordering reversed, $\tilde{y}=3-\texttt{Integer\_Cloud\_Mask}$: $\tilde{y}=3$ is the product's confident-cloud level and $\tilde{y}=0$ its confident-clear level, so the cloud set $\{\tilde{y}\ge2\}$ coincides exactly with the product's $\{$cloudy, probably cloudy$\}$.",
     "A2: label mapping"),

    (r"or $\hat{y}=\tilde{y}^{\rm PP}$ for every PhysPrior-prefixed deep-learning row",
     r"or $\hat{y}=\tilde{y}^{\rm corr}$, the PhysPrior-corrected field, for every PhysPrior-prefixed deep-learning row",
     "A3: yhat^PP"),

    # ---------------------------------------------------------------- A1
    (r"Clear-sky regions exhibit smooth spatial patterns; if the local standard deviation within a 5$\times$5 window is below a threshold $\sigma_{\rm th}$, the pixel is likely clear :",
     r"Clear-sky regions exhibit smooth spatial patterns; if the local standard deviation of the M15 brightness temperature within a 5$\times$5 window is below a threshold $\sigma_{\rm th}$, the pixel is likely clear:",
     "A1: sigma_5 field"),

    (r"When \emph{both} conditions hold for a CLDMSK-labeled ``cloud'' pixel, we flip it to ``clear''.",
     r"$\sigma_5$ is the population standard deviation over the window, with missing values replaced by the patch median; the operator is specified in the supplementary material. When \emph{both} conditions hold for a CLDMSK-labeled ``cloud'' pixel, we flip it to ``clear''.",
     "A1: sigma_5 operator"),

    # ---------------------------------------------------------------- A4
    (r"The third input channel is the DNB (Day/Night Band) static nighttime-lights basemap.",
     r"The third input channel is a season-matched DNB (Day/Night Band) composite, one map per station and season, formed as the per-pixel temporal median of the training and validation scenes whose CLDMSK field is at least $95\%$ clear. It therefore carries the stable artificial-light background and no information from the test period. The channels are DNB, this composite, and M15, in that order.",
     "A4: basemap description"),

    # ---------------------------------------------------------------- A6
    (r"Reduced sensitivity to very thin cirrus ($Z<-35$~dBZ) and attenuation in heavy precipitation are addressed in Section~\ref{sec:limit}.",
     r"Cirrus weaker than the $-33$~dBZ detection threshold, and attenuation in heavy precipitation, are addressed in Section~\ref{sec:limit}.",
     "A6: cirrus threshold wording"),

    # ---------------------------------------------------------------- A5, A18
    (r"where $T \in \mathbb{R}^{4 \times 4}$ is indexed by the four CLDMSK confidence classes in the rows and a binary radar target in the columns; intermediate columns are structural zeros because the radar reference is itself binary. (Equivalently, this is a $2 \times 2$ transition matrix over the cloud/clear mapping, lifted into the 4-class label space; we report the explicit shape for compatibility with the 4-class output head.)",
     r"where $T$ is the $4\times2$ transition matrix $P(\mathrm{radar} \mid \mathrm{CLDMSK\ class})$, embedded in the four-class label space with the two unused columns set to zero so that it applies to the four-class output.",
     "A5: transition matrix shape"),

    (r"Co-Teaching~\cite{han2018co-teaching}; GCE~\cite{zhang2018generalized} ($q{=}0.7$); and Mixup~\cite{zhang2018mixup} ($\alpha{=}0.4$).",
     r"Co-Teaching~\cite{han2018co-teaching} trains two networks that select each other's low-loss samples; its two rows in Table~\ref{tab:methods} are those two peer networks at their selected checkpoints. GCE~\cite{zhang2018generalized} ($q{=}0.7$) and Mixup~\cite{zhang2018mixup} ($\alpha{=}0.4$) follow their original settings.",
     "A18: co-teaching A/B"),

    # ---------------------------------------------------------------- C16, B1
    (r"The dataset comprises $2{,}782$ unique nighttime VIIRS patches from two subtropical South China sites: Changsha ($n{=}901$, urban) and Longmen ($n{=}1{,}881$, suburban), spanning 2019--2026 except for $2021$ at Changsha.",
     r"The dataset is two fixed nighttime scenes, each a $128\times128$ VIIRS field centred on a Ka-band radar: Changsha ($n{=}901$ overpasses, urban) and Longmen ($n{=}1{,}881$, suburban). Every patch is the same field at a different overpass, so the $2{,}782$ samples are two time series rather than a spatial sample. They were extracted from the VIIRS CLDMSK L2 product (Suomi-NPP) and span 2019--2026 except for $2021$ at Changsha.",
     "C16/B1: sampling and platform"),

    # ---------------------------------------------------------------- A7
    (r"and eight trained deep-learning configurations. Two observations organise the deep-learning block.",
     r"and eight trained deep-learning configurations. Unless stated otherwise, each trained row reports the checkpoint with the best radar agreement on the radar-labelled validation patches; the $2\times2$ ablation instead uses a fixed 20-epoch budget, because that agreement is maximised by an all-cloud prediction and saturates within the first epochs. Two observations organise the deep-learning block.",
     "A7: checkpoint protocol"),

    # ---------------------------------------------------------------- A17
    (r"PhysPrior rule (saved for diagnostic)$^{\S}$ & 197 & 69.5\%\,[62.9,\,76.1] & 70.9 & 58.9 & +0.383 \\" + "\n" + r"\midrule" + "\n",
     r"\midrule" + "\n",
     "A17: drop duplicated Table I row"),

    (r"\par\vspace{4pt}" + "\n" + r"{\footnotesize" + "\n" + r"$^{\S}$The ``PhysPrior rule (saved for diagnostic)'' row reports the direct PhysPrior moderate labels under the centre-pixel protocol; it is not a separately trained network." + "\n" + r"\par}" + "\n",
     "",
     "A17: drop the matching footnote"),

    # ---------------------------------------------------------------- A9
    (r"and PhysPrior$+$DL five-seed summary (bottom block).}",
     r"and PhysPrior$+$DL five-seed summary (bottom block). The $\pm$ is one standard deviation over the five seeds.}",
     "A9: define the +/-"),

    # ---------------------------------------------------------------- A10
    (r"with $0\%$ cloud recall against PhysPrior's $76.6\%$.",
     r"with $0\%$ cloud recall against the five-seed mean cloud recall of $76.6\%$.",
     "A10: disambiguate 76.6%"),

    # ---------------------------------------------------------------- A16
    (r"The VIIRS CLDMSK product~\cite{ackerman2010disc} provides binary cloud/clear labels globally",
     r"The VIIRS CLDMSK product~\cite{ackerman2010disc} provides a four-level cloud-confidence mask, distributed and consumed as binary cloud/clear labels",
     "A16: four-level vs binary"),

    # ---------------------------------------------------------------- B4
    (r"VIIRS CLDMSK L2 data are publicly available from the NOAA Comprehensive Large Array-Data Stewardship System (CLASS). The Ka-band radar observations and derived dataset are available from the corresponding author on reasonable request.",
     r"The per-sample prediction records behind every row of Tables~\ref{tab:methods}--\ref{tab:agree}, the $357$ radar-collocated reference labels, the eight station-season basemap composites with their provenance record, the per-patch split manifest, and all analysis code are released at \url{https://github.com/Aokok1/physprior-nighttime-cloud}. VIIRS CLDMSK L2 granules are publicly available from the NOAA Comprehensive Large Array-Data Stewardship System (CLASS). The raw Ka-band radar volumes are not redistributed; the derived reference labels are available to the editor and reviewers on request.",
     "B4: itemised data availability"),

    # ---------------------------------------------------------------- reference [2]
    (r"J.~Huo and X.~Han, ``MODIS cloud mask validation with ground-based radar,'' \textit{Remote Sens.}, vol.~7, no.~8, pp.~10456--10472, 2015.",
     r"J.~Huo and C.~Han, ``Comparison of MODIS cloud mask products with ground-based millimeter-wave radar,'' in \textit{Proc. IGARSS}, 2020, pp.~5329--5332.",
     "ref [2]: correct the citation"),

    # ---------------------------------------------------------------- reference [10]
    (r"``MMFormer: Macro-micro transformer for small-sample classification of Mars hyperspectral image,''",
     r"J.~Feng et al., ``MMFormer: Macro-micro transformer for small-sample classification of Mars hyperspectral image,''",
     "ref [10]: add authors"),
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
