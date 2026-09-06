## Reviewer jTLx

### Weaknesses
* The application of hyperbolic embedding to the problem of missing data is a straightforward application, thus the novelty is somewhat limited. The method doesn't significantly offset the accuracy.
* **(a)** A comparative analysis of the computational complexity between the evaluated models should be included to fully assess efficiency.
* **(b)** Comparison and analysis around the missing content with different methods would provide a stronger case rather than accuracy.
* **(c)** In Figure 4, is the accuracy reported test/train? Additionally, for the ablation study over $k$, the trend between the two actually shows that kNN is doing most of the heavy lifting.
* **(d)** The authors should provide their source code to substantiate their claims and ensure full reproducibility. Discussing the limitations and future scope is also relevant.

**Overall Score:** 3: Weak Reject

---
Thanks for your thoughtful review. Below, we address your concerns.

**Q1.** Novelty is limited, and the accuracy gain does not offset this.

The contribution is the retrieval geometry that feeds a frozen tabular foundation model, not hyperbolic imputation as a stand-alone preprocessor. Accuracy is only one axis; we also look at discrimination, calibration, and behaviour when the context budget is small. Holding TabPFN, the imputer, and the Euclidean features used at inference fixed on TBI-MH103, only the neighbour metric changes:

| Metric | \(k=10\) | \(k=60\) | \(k=110\) | \(k\ge 160\) |
|---|---|---|---|---|
| Euclidean | 68.61 | 82.74 | 82.94 | \(\approx 84\) |
| Hyperbolic | **77.00** | **85.35** | 84.35 | \(\approx 84\) |
| \(\delta\) / \(p\) / wins | **+8.39** / 0.014 / 8 of 10 | **+2.61** / 0.049 / 7 of 10 | +1.41 / 0.25 / 4 of 10 | \(<1\) / n.s. |

The two metrics diverge when \(k\) is small and coincide once the neighbourhood is large enough to saturate the training fold. That pattern is the intended geometric claim: Poincaré retrieval helps when similar cases are scarce, not as a uniform lift on every accuracy table. On the same folds the TabPFN family also improves AUROC versus XGBoost (\(+2.12\), \(p=0.0039\)) and LightGBM (\(+2.30\), \(p=0.0020\)), with lower Brier and ECE (\(p<0.05\)).

**Q2.** Compare computational complexity across the evaluated models.

HICL-TBI does not train a task-specific model. Cost is a linear Poincaré scan over the training fold plus one frozen TabPFN forward pass on at most \(k\) neighbours.

| Method | Training | Per query |
|---|---|---|
| HICL-TBI | none | Poincaré \(O(Nd)\) + one forward pass, context \(\le k\) |
| TabPFN | none | one forward pass on the full training set |
| LoCalPFN | fine-tune on neighbours | one forward pass |
| GBDTs | fit every fold | tree inference |

Relative to full-set TabPFN the extra work is the \(O(Nd)\) retrieval; relative to LoCalPFN and the GBDTs there is no per-fold training. A short complexity paragraph will be added in the revision.

**Q3.** Analyse missingness with different methods, not accuracy alone.

Euclidean reconstruction RMSE is not the right criterion here. TBI-MH103 is tree-like in the sense used in the paper (\(\delta_{\mathrm{rel}}\): \(0.417\to 0.069\); radius–severity Spearman \(\rho=+0.59\)), so neighbours are ranked with the Poincaré distance \(\delta_D\). A flat mean or Euclidean \(k\)NN fill pulls completed points toward the origin and flattens that hierarchy. Fréchet imputation uses the Riemannian centre of mass so the completed matrix stays on the same manifold that retrieval uses.

The controlled comparison therefore changes only the imputer, with retrieval and TabPFN held fixed (MH103, \(K=5\)): Euclidean \(k\)NN 85.06 versus Fréchet **86.15**. We treat this as a downstream, same-protocol check, not as an RMSE contest, and will spell out that distinction in the revision.

**Q4.** Is Figure 4 train or test, and is \(k\)NN doing most of the work?

Figure 4 is **test** accuracy under stratified 10-fold CV; the caption will say so. \(k\)NN is only the search procedure. The distance decides which rows enter the context, and that choice is what we isolate. At matched \(k=10\), Poincaré beats \(L_2\) by \(+8.39\) (\(p=0.014\), 8 of 10 folds). At \(k=60\) the gap is \(+2.61\) (\(p=0.049\)). Switching Euclidean retrieval from fixed \(k=60\) to adaptive \(k\) adds only \(+0.40\). The metric, not the mere presence of a nearest-neighbour step, accounts for the difference.

**Q5.** Provide source code, and discuss limitations and future work.

Anonymous code is at https://anonymous.4open.science/r/HICL-TBI-B691 (implementation and evaluation harness; no clinical records). De-identified fold CSVs are available from the corresponding author on reasonable request. The main limitations are the single-centre MH103 cohort and the fixed curvature \(c=1\). We will expand the limitations and future-work discussion in the revision.

---

## Reviewer KRez

### Weaknesses
* **Small gains and lack of statistical testing:** The gains are small and no statistical test is provided. HICL-TBI ties LoCalPFN on one dataset, loses to plain TabPFN in the ODC-TBI 4-class setting, and has only a small gain in the 8-class setting. Since the paper uses 10-fold cross-validation, it should report variation across folds and paired tests.
* **Confounded improvements:** The two main changes are not separated. HICL-TBI changes both the distance metric and the context size. Without Euclidean retrieval with adaptive context size and hyperbolic retrieval with fixed context size, the source of the gain is unclear.
* **Missing critical hyperparameters:** The paper does not report the curvature, the minimum and maximum context sizes, or a clear tuning process. These choices are central to the method and must be stated.
* **Limited baselines and metrics:** Strong tabular baselines such as XGBoost, LightGBM, and CatBoost are missing. For imbalanced clinical prediction, the paper should also report macro-F1, balanced accuracy, and AUROC.
* **Inconsistent values:** Several values are inconsistent:
  * The text reports 35.96% for the 8-class result, while Table 2 reports 36.33%.
  * ODC-TBI has 34 features in the text but 27 in Table 1.
  * TBI-MH103 has 68 features in Section 4.1 but 64 in Table 1.
  * The TabNet specificity value "8970" and the duplicate or malformed references should also be corrected.
* **Incomplete reproducibility and ethics info:** TBI-MH103 is a new clinical dataset, but the paper gives no ethics approval, consent, anonymization, or data-sharing statement. There is also no code availability statement.

**Overall Score:** 3: Weak Reject

---
Thanks for your thoughtful review. Below, we address your concerns.

**Q1.** Gains are small and untested; report fold variance and paired tests.

The revision reports mean \(\pm\) std over the same ten folds as Table 2, together with 95% CIs and paired Wilcoxon tests. We do not claim a uniform win on every dataset.

| Dataset | HICL-TBI | TabPFN | LoCalPFN |
|---|---|---|---|
| Pilot | \(84.00\pm 14.67\) | \(82.09\pm 14.81\) | \(84.00\pm 15.67\) |
| TBI-MH103 | \(\mathbf{86.15\pm 5.77}\) | \(83.35\pm 5.61\) | \(84.74\pm 4.72\) |
| ODC 4-class | \(98.34\pm 0.74\) | \(\mathbf{98.59\pm 0.77}\) | \(98.04\pm 0.78\) |
| ODC 8-class | \(\mathbf{36.33\pm 3.23}\) | \(35.48\pm 2.38\) | \(35.43\pm 2.35\) |

HICL-TBI is tied or best among the training-free methods on three of four tasks. ODC 4-class sits at ceiling for every strong model; the 0.25-point gap versus TabPFN is not significant (\(p=0.77\)). On MH103, matched-\(k\) hyperbolic versus Euclidean retrieval is significant at \(k=60\) (\(p=0.049\)). The Pilot \(\pm 15\) band is the small-\(N\) variability you noted, not hidden instability of the method. Full per-fold tables will appear in the revision.

**Q2.** Metric and adaptive \(k\) are confounded; separate them.

We separate the two choices on TBI-MH103 with identical folds, the same imputer, and frozen TabPFN.

| | Fixed \(k\) | Adaptive \(k\) |
|---|---|---|
| Euclidean | \(83.15\pm 4.93\) (\(k=60\)) | \(83.55\pm 4.83\) |
| Hyperbolic | **85.35** (\(k=60\)) / **77.00** (\(k=10\)) | **86.15** |

Adaptive \(k\) on Euclidean retrieval adds only \(+0.40\). At matched \(k\), hyperbolic retrieval adds \(+2.61\) at \(k=60\) (\(p=0.049\), 7 of 10 folds) and \(+8.39\) at \(k=10\) (\(p=0.014\), 8 of 10). The distance metric, not the adaptive radius, is the dominant factor. The revision will also state \(k_{\max}\) explicitly so the adaptive rule cannot collapse to the full training set.

**Q3.** Curvature, \(k_{\min}/k_{\max}\), and the tuning process are missing.

These values are now fixed in the method statement. They were used for all main results unless a figure caption names a diagnostic \(k\).

| Symbol | Role | Value |
|---|---|---|
| \(c\) | curvature (\(\kappa=-c\)) | **1.0** |
| \(r^\ast\) | de-saturation (Fig. 5) | **0.9** |
| \(K_{\mathrm{imp}}\) | Fréchet neighbours | 5 |
| \(n_{\mathrm{iter}}\), \(\mathrm{tol}\) | Einstein midpoint | 10, \(10^{-4}\) |
| \(m_{\mathrm{ref}}\), \(\alpha\) | bandwidth | 10, 3.0 |
| \(k_{\min}\), \(k_{\max}\) | clip on \(k^\ast\) | 10, bounded |
| \(N_{\mathrm{ens}}\) | frozen TabPFN | 18 |

A short sensitivity check over \(c\), \(\alpha\), \(k_{\max}\), and \(r^\ast\) will be included in the revision.

**Q4.** Add strong tabular baseline such as XGBoost, LightGBM, and CatBoost.

The revision adds XGBoost, LightGBM, and CatBoost, and reports macro-F1, balanced accuracy, and AUROC (AUPRC, Brier, and ECE in the supplement). These are trained, per-fold supervised baselines; HICL-TBI remains training-free. On the accuracy metric of Table 2 they do not change the ranking of the two settings where HICL-TBI is ahead:

| | HICL-TBI | XGBoost | CatBoost | LightGBM | TabPFN |
|---|---|---|---|---|---|
| TBI-MH103 acc. | **86.15** | 84.95 | 84.74 | 84.15 | 83.35 |
| ODC 8-class acc. | **36.33** | 32.71 | 34.34 | 32.51 | 35.48 |

On MH103 the TabPFN family also ranks and calibrates at least as well as the GBDTs (AUROC \(+2.1\)–\(2.3\) versus XGBoost/LightGBM, \(p<0.01\); lower Brier and ECE). The extra metrics therefore complement, rather than overturn, the main comparison.

**Q5.** Inconsistent values: 35.96 vs 36.33, feature counts, TabNet 8970, and a malformed reference.

We thank you for catching these. The 8-class accuracy is **36.33** (Table 2); 35.96 in the text is a transcription error and will be corrected. Section 4.1 lists raw fields (MH103 68, ODC 34); Table 1 lists modelled features after dropping unused or outcome-leaking columns (64, 23, and 16 on the pilot). Both counts will be stated together. TabNet specificity is \(8970\to\mathbf{89.70}\). The duplicate Bruschetta reference will be merged.

**Q6.** Missing ethics, data-sharing, and code-availability statements.

TBI-MH103 was collected under project **B2023-BKA-09** (HUST \(\times\) Military Hospital 103; PI = last author). Records are fully de-identified and were approved for research use. Data are available from the corresponding author on reasonable request. Anonymous code (implementation and evaluation harness; no clinical records) is at https://anonymous.4open.science/r/HICL-TBI-B691. These statements will appear in the revision.

---

## Reviewer KcKF

### Weaknesses and Required Revisions
* **Potential leakage and preprocessing ambiguity:** Imputation, standardization, de-saturation, radius selection, and hyperparameter choices must be fit inside each training fold. Clarify whether labels or outcome-derived variables influence retrieval or preprocessing, and provide a complete feature list and missingness handling protocol. The claim of training-free inference does not remove leakage risk.
* **Small and non-independent clinical cohorts:** The 102-subject cohort and single-hospital MH103 cohort are underpowered for four-class claims. Report fold-level confusion matrices, confidence intervals, calibration (Brier/ECE), AUROC/AUPRC, and class prevalence. Add patient-level/site-level external validation or substantially temper clinical deployment language.
* **Baseline and statistical fairness:** TabPFN, LoCalPFN, and HICL use different context construction and possibly different preprocessing. Include repeated nested CV, paired statistical tests, tuned Euclidean metrics, ordinal models, and a simple severity-aware baseline. Improvements of 0.2–1.4 percentage points need uncertainty estimates.
* **Incomplete method specification:** Define the exact Poincaré distance/curvature, Fréchet-mean solver and stopping criterion, density estimator, adaptive context-size rule, TabPFN version/context limit, and computational cost. The text alternates between $K=5$, $K=60$, and reference-neighbor settings without a single unambiguous protocol.

### Overall Assessment
The framework is an interesting hypothesis and may be publishable after rigorous leakage-controlled validation and clearer geometry/statistical analysis. In its current form, the clinical conclusions are premature.

**Overall Score:** 3 - Weak reject

---
Thanks for your thoughtful review. Below, we address your concerns.

**Q1.** Preprocessing must be train-fold only; labels must not enter retrieval. Training-free does not remove leakage risk.

We agree that “training-free” does not, by itself, rule out leakage. The protocol is: every transform is estimated on the training fold only, and labels never enter imputation, scaling, de-saturation, bandwidth, or neighbour ranking.

| Step | Fit on train | Test fold |
|---|---|---|
| Column statistics | yes | apply |
| Fréchet imputer | training pool | query that pool only |
| Scaler / \(r^\ast=0.9\) | train \(\mu,\sigma\), max radius | transform |
| \(\sigma_u\), retrieval | train distances / candidates | train indices only |

Labels appear only as the targets of already-retrieved *training* rows, which is standard in-context learning. A complete feature list and the missingness protocol will be placed in the appendix.

**Q2.** Cohorts are small and single-site; report prevalence, calibration, and CIs, and temper deployment claims.

We agree that the pilot (\(N=102\)) is underpowered for a four-class clinical claim and treat it as a small-data stress test. MH103 is a single-centre cohort; we make no deployment claim. ODC-TBI is the larger, external check.

| Cohort | \(N\) | Missing | Prevalence (\%) |
|---|---|---|---|
| Pilot | 102 | 3.16 | 28.4 / 8.8 / 37.3 / 25.5 |
| TBI-MH103 | 504 | 5.39 | 10.9 / 58.1 / 20.0 / 10.9 |
| ODC 4-class | 2531 | 8.41 | 48.3 / 25.0 / 9.3 / 17.4 |
| ODC 8-class | 2015 | 8.10 | 4.3 / 4.9 / 14.0 / 4.6 / 21.7 / 17.3 / 17.3 / 15.9 |

On TBI-MH103, HICL-TBI (10-fold mean \(\pm\) std) gives AUROC \(93.82\pm 2.01\), AUPRC \(83.48\pm 3.95\), Brier \(0.24\pm 0.06\), and ECE \(0.09\pm 0.03\). Fold-level confusion matrices and 95% CIs will be included in the revision.

**Q3.** Protocols are unequal; small gains of 0.2–1.4 points need uncertainty.

TabPFN, Euclidean retrieval, and HICL-TBI now share the same folds, imputer, scaler, and frozen TabPFN weights; only the neighbour set changes. We additionally report a tuned Euclidean retrieval arm and per-fold GBDT baselines. Every comparison in the revision is mean \(\pm\) std, a 95% CI, and a paired Wilcoxon test over the ten folds. ODC 4-class is \(\approx 98\%\) for every strong model, so the 0.2–0.3 point gaps there should not be read as a method ranking.

**Q4.** Specify the full method; \(K=5\), \(K=60\), and \(m_{\mathrm{ref}}\) are ambiguous.

There is one evaluation protocol; the former \(K\) symbols named different quantities and will be disambiguated. Curvature \(c=1.0\). Poincaré distance
\(\delta(u,v)=(1/\sqrt{c})\,\mathrm{arccosh}\bigl(1+2c\|u-v\|^2/((1-c\|u\|^2)(1-c\|v\|^2))\bigr)\).
The exponential map uses a train-fit de-saturation \(r^\ast=0.9\). Imputation: Einstein midpoint, \(n_{\mathrm{iter}}=10\), \(\mathrm{tol}=10^{-4}\), \(K_{\mathrm{imp}}=5\). Adaptive context: \(\sigma_u=\delta(u,v_{m_{\mathrm{ref}}})\), \(m_{\mathrm{ref}}=10\), \(\alpha=3.0\), \(k^\ast\in[10,k_{\max}]\). Inference: frozen TabPFN with \(N_{\mathrm{ens}}=18\); cost \(O(Nd)\) plus one forward pass. The \(K=60\) curves in Fig. 5 are a diagnostic sweep, not the adaptive rule used in Table 2.

---

## Reviewer xkTX

### Weaknesses
* **Headline claims contradict Table 2:** HICL-TBI ties LoCalPFN on the pilot cohort and loses to plain TabPFN on ODC-TBI 4-class, yet the abstract and conclusion claim broad superiority. The supportable claim is narrow (gains concentrate on the fine-grained 8-class task), and the paper should make that claim instead.
* **Sub-1.5 point winning margins without variance:** The flagship 8-class and TBI-MH103 wins are reported as point estimates under 10-fold CV with no standard deviations or significance, and Figure 5 shows large fold variance, so the wins are not interpretable as stated.
* **Unspecified curvature:** The curvature is never specified, despite parameterizing every distance and the entire outlier-separation argument, which is a core reproducibility gap.
* **Imputation claim relies on minimal data:** The imputation contribution rests on one number pair (86.15 vs 85.06, $K=5$, one cohort, no variance).
* **Inconsistent dataset descriptions:** Feature counts differ between Table 1 and the text for both TBI-MH103 and ODC-TBI.
* **Neighbor-selection effect vs. geometry:** The benefit is a neighbor-selection effect only (Euclidean features are restored before inference), so the evaluation needs a controlled same-$k$ hyperbolic-vs-Euclidean selection comparison to attribute the gain to geometry rather than to the adaptive context size.

**Overall Score:** 3: Weak Reject

---
Thanks for your thoughtful review. Below, we address your concerns.

**Q1.** Abstract and conclusion overclaim relative to Table 2.

We agree that the abstract and conclusion must follow Table 2 rather than a uniform-superiority claim. The revision states a tie on the pilot (84.00), a near-tie on saturated ODC 4-class (98.34 vs 98.59), and the two settings that favour HICL-TBI: MH103 (86.15) and 8-class (**36.33** vs 35.48 / 35.43). Those last two tasks are the hierarchical-severity cases the method is designed for. On MH103, AUROC also favours the TabPFN family over the GBDTs (\(+2.1\)–\(2.3\), \(p<0.01\)).

**Q2.** Flagship wins lack variance and significance; Figure 5 shows large fold spread.

The revision reports mean \(\pm\) std, 95% CIs, paired Wilcoxon tests, and fold-win counts over the same ten folds as Table 2. The Mild-class spread in Fig. 5 tracks class prevalence; the per-class gaps at the diagnostic \(K=60\) remain (Mod. 96.4 vs 92.9, Sev. 85.3 vs 81.4, V.Sev. 97.1 vs 94.2). When the accuracy delta is small, AUROC, AUPRC, Brier, and ECE are the more stable reading, and those will be tabulated with the same fold protocol.

**Q3.** Curvature is never specified.

All distances in the paper use \(c=1.0\) (\(\kappa=-1\)). The de-saturation radius in Fig. 5 is \(r^\ast=0.9\) and will be named in the method. A short sensitivity check over \(c\) and \(r^\ast\) will be included in the revision.

**Q4.** Imputation rests on one pair (86.15 vs 85.06) with no variance.

That pair is a controlled swap, not a full imputation benchmark: retrieval, \(K=5\), and TabPFN stay fixed, and only the imputer changes (Euclidean \(k\)NN 85.06 vs Fréchet **86.15** on MH103). Gromov \(\delta\) motivates hyperbolic *retrieval*; Fréchet imputation is there so the completed matrix remains on that manifold. We do not claim state-of-the-art RMSE reconstruction, and we will mark the comparison as a single-cohort, same-protocol check.

**Q5.** Feature counts differ between Table 1 and the text.

The two numbers are different stages of the same pipeline and will be stated together. Section 4.1 counts raw fields (MH103 68, ODC 34). Table 1 counts modelled features after unused or outcome-leaking columns are dropped (64, 23, and 16 on the pilot). ODC 4-class and 8-class use different eligible subsets because unresolved GOSE cases are dropped for the 8-class task.

**Q6.** Need a same-\(k\) hyperbolic vs Euclidean comparison; geometry may only select neighbours.

We agree that geometry acts only through the retrieved indices: Euclidean features are restored before the TabPFN forward pass. The attribution test is therefore matched-\(k\) neighbour selection, with the imputer and the frozen model held fixed:

| | \(k=10\) | \(k=60\) | \(k=110\) | \(k\ge 160\) |
|---|---|---|---|---|
| Euclidean | 68.61 | 82.74 | 82.94 | \(\approx 84\) |
| Hyperbolic | **77.00** | **85.35** | 84.35 | \(\approx 84\) |
| \(\Delta\) / \(p\) / wins | **+8.39** / 0.014 / 8 of 10 | **+2.61** / 0.049 / 7 of 10 | +1.41 / 0.25 / 4 of 10 | \(<1\) / n.s. |

Switching Euclidean retrieval from fixed \(k=60\) to adaptive \(k\) adds only \(+0.40\). The gain is geometry-driven neighbour selection, and it is largest when the context budget is scarce.
