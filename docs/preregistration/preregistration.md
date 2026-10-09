# GeoShift — pre-registration

**Paper 2 of the geochemistry × machine-learning programme.**
Registered 25 September 2026, before any model is fitted.
Methodology of record: `02-methodology/_LOCKED/methodology-v5-LOCKED-2026-09-25.md`.
Target venue: Earth System Science Data.

Authors: Shaho Hassen (Salahaddin University) · Dashti Arif Sulaiman.

This document fixes the hypotheses, the element set, the fold assignments, the
acceptance criteria and the analysis plan. Everything it names is frozen and hashed
in `lock-manifest-prereg-2026-09-25.csv`. No result has been computed for any task.

---

## 1 Hypotheses and directional predictions

**H1 — censoring encodes analytical provenance.** A classifier reading only the
censoring mask identifies the source survey above the best single-element rule.
*Refuted if* balanced accuracy does not exceed the best-single-element null on the
held-out fold.

**H2 — provenance-blocked validation is materially more pessimistic than random
validation.** Task 1 performance under leave-one-survey-out falls below random
*k*-fold by more than the spread across random-fold repetitions.
*Refuted if* the LOSO and random *k*-fold intervals overlap, or if LOSO is better.

**H3 — the penalty scales with detection-limit disparity.** Within the NGSA↔NASGL
pair, per-element Task 1 degradation increases with log Δ.
*Refuted if* the bootstrap interval for the slope on log Δ contains zero.

**H4 — the effect survives geography.** On Tellus, where site, sample and custodian
are identical across two analytical routes, route is identifiable from the mask above
the best single-element rule, and the paired censoring indicators disagree.
*Refuted if* route accuracy does not exceed that null. **Partially observed
already**: the paired disagreement is measured at 36 of 47 elements above 5%, median
18.45%; the classifier is not yet fitted.

**H5 — censoring shift is separable from geochemical affinity.** The censoring
balance and the affinity-only balance diverge across protocols.
*Refuted if* the two balances move together, in which case the shift is geochemical
and the paper's claim narrows to that.

**Stated in advance:** H2 and H3 refuted while H1 and H4 hold would mean censoring
carries provenance without costing cross-region accuracy. That is a publishable
negative result for a benchmark paper and will be reported as the headline if it
occurs. No hypothesis will be dropped from the report.

---

## 2 Element set — frozen

39 primary-core elements, in `primary-core-elements.csv`: the 44 reported by all
three observational surveys, less Li, In, Te, Tl and Se, which are aqua regia in
GEMAS and NGSA against near-total in NASGL with cross-survey median ratios of
1.8–3.5× (methodology §2.1). Those five run in Task 5 only. Fe and Hg are retained
unrestricted, and Fe is **not** excluded as a normaliser.

Task 3 runs on the **34** of those 39 for which both NGSA and NASGL publish a
detection limit (`lod-disparity-delta.csv`).

---

## 3 Fold assignments — frozen, seed 20260925

| Protocol | File | Folds | Note |
|---|---|---|---|
| Random *k*-fold | `folds-random-kfold.csv` | 10, survey-stratified | all 10 contain all three surveys |
| Leave-one-survey-out | `folds-loso.csv` | 3 | GEMAS 0, NGSA 1, NASGL 2 |
| Spatial blocking | `folds-spatial-block.csv` | 10, from 706 2° cells | all 10 contain all three surveys |
| Environmental blocking | `folds-environmental.csv` | 10, k-means in standardised log-element space | **only 5 of 10 contain all three surveys** |
| NNDM | `nndm-target-distributions.csv` | LOO | see §3.2 |

Fold assignment uses coordinates, survey labels and predictor values, never any
task's target, so computing it on the full table is not leakage. It is nonetheless
recorded here so that no later re-draw is possible.

### 3.1 Environmental blocking partially separates surveys, and that is reported

Clusters 2, 3, 6 and 8 are GEMAS-dominated and cluster 5 contains no GEMAS, so only
5 of 10 environmental folds are three-survey. Blocking in element space therefore
recovers survey identity to a substantial degree on its own. This is evidence for
H1 by a second route and is reported as such, not corrected away.

### 3.2 NNDM-geographic is vacuous on LOSO folds — declared in advance

Measured nearest-neighbour distances from each held-out survey to the training
surveys (`nndm-target-distributions.csv`):

| Held out | median NN distance | range |
|---|---|---|
| GEMAS | 5,475 km | 4,531–6,906 km (5th–95th) |
| NGSA | 11,922 km | 10,977–12,844 km |
| NASGL | 6,525 km | 5,173–7,647 km |

NNDM matches a nearest-neighbour distance distribution *below an autocorrelation
range*. The range for soil geochemistry is tens to low hundreds of kilometres. At
5,000–12,000 km no training point lies within any plausible range of any test point,
so no exclusion is required and NNDM collapses to ordinary leave-one-out.
**NNDM-geographic is therefore reported for random *k*-fold and within-survey splits
only, and is declared inapplicable to LOSO folds rather than run and presented as if
it were informative.** This is registered before the fact.

### 3.3 NNDM-mask is not vacuous, with one caveat stated now

Jaccard nearest-neighbour distance on the censoring mask:

| Held out | median | mean | max |
|---|---|---|---|
| GEMAS | 0.000 | 0.023 | 0.500 |
| NGSA | 0.375 | 0.347 | 0.750 |
| NASGL | 0.500 | 0.383 | 0.800 |

NGSA and NASGL are separated in mask space; GEMAS is not. **The GEMAS value is
partly an artefact of metadata availability**: GEMAS publishes censoring flags for 15
of 53 elements, so many of its core-element mask columns are all-zero and its mask
sits at zero distance from any other sparse mask. GEMAS mask distances will not be
interpreted as evidence of analytical similarity, and any H1 result involving GEMAS
will be reported alongside its flag coverage.

---

## 4 Acceptance criteria — fixed before fitting

Tasks 2 and 4 report three nulls: majority class, mask density, and the **best
single-element rule**, the element whose censoring indicator alone best separates the
classes, selected on the training fold. The single-element rule is binding. Measured
on Tellus already: balanced accuracy 0.996 (Hg), 0.974 (Ag), 0.876 (W), 0.852 (Bi),
0.841 (Ti). A learned classifier that does not exceed the best single-element rule is
reported as having demonstrated nothing on that task.

---

## 5 Analysis plan

- **Preprocessing** fitted within training folds only; imputation per survey and,
  in the controlled arm, per route. ilr on the frozen Paper 1 basis.
- **Metrics** normalized RMSE, MAPE, R² for Task 1; balanced accuracy for Tasks 2
  and 4. Per-fold values reported in full alongside any aggregate.
- **Intervals** nested cross-validation. With three non-exchangeable LOSO folds the
  aggregate interval is indicative, not calibrated, and **no significance claim is
  made on three folds**, inherited claims included.
- **Task 3** one fold pair, 34 element-level points, element-level bootstrap on the
  slope against log Δ, with residual within-pair dependence acknowledged and not
  modelled. No generality claim over laboratories.
- **Task 4** within-sample paired comparison at n = 5,643; conventional paired
  inference applies here and nowhere else.
- **Confound controls** stratified label permutation on the 35 multi-survey
  Köppen × GLiM strata; fold-internal mask regression after Snoek et al.; paired
  limits of agreement on censoring indicators, not concentrations.
- **Multiplicity** Benjamini–Hochberg within each task family, applied only where a
  test is admissible under the rules above.

---

## 6 Deviation policy

Any departure from this document is recorded in a dated deviations log in this
folder, stating what changed, why, and whether it was decided before or after seeing
the affected result. Deviations decided after seeing a result are labelled as such in
the manuscript. The methodology file may be amended only with a new version number
and a new lock entry.

**Deviation already required (v6):** methodology §3.3 anticipated that NNDM-mask
might lack a target but did not anticipate §3.2 above — that NNDM-*geographic* is
vacuous on LOSO folds. This was discovered when the target distributions were
computed for this registration, before any model was fitted, and is recorded here.

---

## 7 What has already been computed, and what has not

**Computed** (dataset construction and design audits only): censoring masks and
limits for all four sources; the paired route contrast on Tellus; the element
crosswalk; the Köppen × GLiM strata and their permutation power; the detection-limit
table and Δ; the fold assignments and NNDM target distributions; the
best-single-element nulls.

**Not computed:** every task. No model has been fitted. No value of any metric in §5
exists for any protocol.

---

## 8 Where this is deposited

This folder is the authoritative copy. A timestamped external deposit should be made
before the first model run, to OSF (osf.io, free registration, DOI on registration)
or Zenodo (zenodo.org, free, DOI), uploading this file together with the frozen
artefacts and the lock manifest. Both require an account, so the deposit is made by
the first author; the manifest hashes below let anyone verify that what was deposited
is what is described here.
