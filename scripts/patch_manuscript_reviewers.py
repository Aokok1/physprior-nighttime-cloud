"""Address the GRSL-02086 review comments that the current manuscript still misses.

From the decision letter (2026-08-19, EiC Ronny Hänsch):

  Associate Editor ... "Table 1 is missing from the document."
  Reviewer 1 ....... "all Tables are offset by 1 (Table II is Table I)..."
  Reviewer 2 ....... 1. advantages of PhysPrior over existing approaches
                     2. Where is TABLE I?
                     3. What is "No-leakage split"?
                     4. remove the script name from the main text
                     5. detailed analysis for the per-station/per-season section
                     6. what the bold font means
                     7. avoid unclear abbreviations such as "pp"
                     8. potential of foundation models (HyperSIGMA, RingMo)

Comments AE, R1, R2-2 and R2-3 are already resolved. This patch handles
R2-1, R2-4, R2-6, R2-7 and R2-8; R2-5 is handled here too.

Run:  python -X utf8 scripts/patch_manuscript_reviewers.py
Then: python -X utf8 scripts/verify_manuscript_numbers.py  and recompile.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"

R = [
    # ---- R2-7: define "pp" at its first use -----------------------------
    (r"reach $70.36\% \pm 2.28$~pp accuracy, versus",
     r"reach $70.36\% \pm 2.28$~percentage points (pp) accuracy, versus",
     "R2-7: define pp in the abstract"),

    # ---- R2-4: no script or artifact names in the main text -------------
    (r"which is the lowest-confidence clear class and the assignment used by the implementation in \texttt{methods/physical\_prior.py}.",
     r"which is the lowest-confidence clear class and the assignment made by the reference implementation.",
     "R2-4: remove script name in Sec. II-A"),

    (r"every confusion-matrix entry and bootstrap $95\%$ interval is computed by \texttt{rebuild\_paper\_tables.py} from the per-sample prediction records of the constituent runs.",
     r"every confusion-matrix entry and bootstrap $95\%$ interval is computed from the per-sample prediction records of the constituent runs.",
     "R2-4: remove script name in Sec. III-B"),

    (r"All metrics and $95\%$ bootstrap confidence intervals are computed by \texttt{rebuild\_paper\_tables.py} from \texttt{output/paper\_tables\_grp.json}, which aggregates the per-sample prediction records named for each row.",
     r"All metrics and $95\%$ bootstrap confidence intervals are computed from the released per-sample prediction records of the runs named for each row.",
     "R2-4: remove script and file names in the Table I caption"),

    (r"\footnote{These fractions are computed directly from the frozen prediction files and exported by \texttt{rebuild\_paper\_tables.py}; see the \texttt{physprior\_flip\_analysis} entry of \texttt{output/paper\_tables\_grp.json}.}",
     r"\footnote{These fractions are computed directly from the per-sample prediction records of the two label fields.}",
     "R2-4: remove script and file names in the flip footnote"),

    (r"The numbers are produced by \texttt{rebuild\_paper\_tables.py} from the per-sample prediction records of this ablation.",
     r"The numbers come from the per-sample prediction records of this ablation.",
     "R2-4: remove script name in the ablation section"),

    # ---- R2-6: state what bold means ------------------------------------
    (r"Methods with ``* '' are trivial baselines; ``\#'' denotes a post-hoc oracle sensitivity benchmark rather than a fair comparator.}",
     r"Methods with ``* '' are trivial baselines; ``\#'' denotes a post-hoc oracle sensitivity benchmark rather than a fair comparator. Bold marks the best value in each metric column.}",
     "R2-6: explain bold in the Table I caption"),

    (r"The flipping analysis that follows refers to this confusion matrix.}",
     r"The flipping analysis that follows refers to this confusion matrix; bold marks the two error cells.}",
     "R2-6: explain bold in the Table II caption"),

    (r"the displayed one-decimal figures differ by up to $\pm 0.05$~pp from them.}",
     r"the displayed one-decimal figures differ by up to $\pm 0.05$~pp from them. Bold marks the best cell.}",
     "R2-6: explain bold in the Table V caption"),

    # ---- R2-1: advantages over existing approaches ----------------------
    (r"Transfer of a fixed rule across sites, rather than a development-set selection, is therefore what this result demonstrates.",
     r"Transfer of a fixed rule across sites, rather than a development-set selection, is therefore what this result demonstrates. Prior remedies fall into two families: threshold-based detectors replace CLDMSK with a competing mask, and noise-robust training objectives treat the label noise as unstructured. PhysPrior does neither. It edits the label field once, with a physical sign, so the corrected labels keep the product's four-class spatial structure and can be handed unchanged to any downstream training method.",
     "R2-1: advantages over existing approaches"),

    # ---- R2-5: per-station / per-season analysis ------------------------
    (r"The winter value is consistent with reduced cloud--surface thermal contrast but could also reflect threshold mismatch, cloud-type differences, or sampling variability; the seasonal results are exploratory rather than climatologically representative.",
     r"Restricting the same model to Longmen alone, where site is fixed and only season varies, the spread persists: $69.4\%$ in summer ($n{=}72$) against $82.5\%$ in autumn ($n{=}40$), so the seasonal pattern is not only a station contrast. All $168$ Longmen patches come from a single year, however, so season and year remain inseparable there. The winter value is consistent with reduced cloud--surface thermal contrast but could also reflect threshold mismatch, cloud-type differences, or sampling variability; the seasonal results are exploratory rather than climatologically representative.",
     "R2-5: per-season analysis within one station"),

    # ---- R2-8: foundation models ----------------------------------------
    (r"Recent representation-learning advances~\cite{yang2026thsgr,yang2024iter,feng2026mmformer} address \emph{representation} rather than \emph{systematic cloud-mask label bias}; combining them with physics-guided correction is a natural extension.",
     r"Recent representation-learning advances~\cite{yang2026thsgr,yang2024iter,feng2026mmformer} and remote-sensing foundation models pretrained by masked image modelling~\cite{sun2023ringmo,wang2025hypersigma} address \emph{representation} rather than \emph{systematic cloud-mask label bias}. They would change the backbone, not the labels, so coupling them to physics-guided correction is a natural extension; the present training pool of $1{,}971$ patches is far too small to pretrain or fine-tune such a model, and the corrected labels would be the natural supervision if a larger pool were assembled.",
     "R2-8: foundation models"),

    # ---- R2-8: two new references ---------------------------------------
    (r"\bibitem{ackerman2010disc}",
     r"\bibitem{sun2023ringmo}"
     "\n" r"X.~Sun et al., ``RingMo: A remote sensing foundation model with masked image modeling,'' \textit{IEEE Trans. Geosci. Remote Sens.}, vol.~61, pp.~1--22, 2023."
     "\n\n" r"\bibitem{wang2025hypersigma}"
     "\n" r"D.~Wang et al., ``HyperSIGMA: Hyperspectral intelligence comprehension foundation model,'' \textit{IEEE Trans. Pattern Anal. Mach. Intell.}, vol.~47, no.~8, pp.~6427--6444, 2025."
     "\n\n" r"\bibitem{ackerman2010disc}",
     "R2-8: add RingMo and HyperSIGMA references"),
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
