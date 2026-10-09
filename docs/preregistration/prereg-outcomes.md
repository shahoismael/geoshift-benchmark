# GeoShift — pre-registered outcomes

**Companion to `preregistration.md`, which is frozen and unchanged.** That document
fixed five hypotheses and their refutation conditions on 25 September 2026, before any
model was fitted. This document records what the runs returned, scored against those
conditions as written. Nothing here amends a hypothesis, a criterion or a refutation
condition; where a hypothesis is refuted it is recorded as refuted.

Runs executed 25 September – 7 October 2026, harness 7.0.0, seed 20260925, on the fold
files hashed in `lock-manifest-prereg-2026-09-25.csv`. Methodology of record for these
outcomes: `02-methodology/methodology-section-final.md` v7.2.

---

## Verdicts

| | Hypothesis | Verdict |
|---|---|---|
| H1 | Censoring encodes analytical provenance | **HOLDS** |
| H2 | Provenance-blocked validation is materially more pessimistic | **HOLDS** |
| H3 | The penalty scales with detection-limit disparity | **REFUTED** |
| H4 | The effect survives geography (Tellus, paired routes) | **HOLDS** |
| H5 | Censoring shift is separable from geochemical affinity | **UNTESTED AS REGISTERED** |

One of five refuted, one untested as registered. The pre-registration stated in advance that a mixed outcome would
be reported as the headline rather than reframed, and that no hypothesis would be
dropped. Both commitments are kept below.

---

## H1 — censoring encodes analytical provenance. HOLDS.

*Refutation condition: balanced accuracy does not exceed the best-single-element null
on the held-out fold.*

Survey identity predicted from the censoring mask alone, every fold of every eligible
protocol (`task2_*.csv`):

| Protocol | Folds | Balanced accuracy | Binding null | Beats null |
|---|---|---|---|---|
| random_k3 | 3 | 0.9636 – 0.9726 | 0.7251 – 0.7424 | 3 of 3 |
| random_k10 | 10 | 0.9571 – 0.9812 | 0.7114 – 0.7618 | 10 of 10 |
| spatial_k3 | 3 | 0.9613 – 0.9709 | 0.7004 – 0.8005 | 3 of 3 |
| environmental_k3 | 3 | 0.9531 – 0.9723 | 0.7126 – 0.7767 | 3 of 3 |

Margin over the binding null is never below 0.16 balanced-accuracy points. The
stratified permutation control (1,000 pooled draws on Köppen × GLiM strata) returns
observed 0.9693 against a permutation maximum of 0.5614, *p* = 0.000999, so the result
does not reduce to climate or lithology.

**Pairwise form, Task 2p — passes on two of three pairs and fails on one**
(`task2p_pairwise.csv`). Reported as a partial failure, not folded into the H1 pass:

| Pair | Balanced accuracy | Best-single-element null | Margin | Criterion |
|---|---|---|---|---|
| GEMAS – NGSA | 0.99810 | 0.99240 | +0.00570 | met |
| GEMAS – NASGL | 0.99331 | 0.99331 | 0.00000 | **not met — ties the null exactly** |
| NGSA – NASGL | 0.95770 | 0.89472 | +0.06298 | met |

On GEMAS – NASGL a single element's censoring indicator does all the work the learned
classifier does. The pre-registered reading applies without softening: on that pair the
classifier demonstrated nothing.

---

## H2 — provenance-blocked validation is materially more pessimistic. HOLDS.

*Refutation condition: the LOSO and random k-fold intervals overlap, or LOSO is better.*

Matched at K = 3 so that the blocking axis is the only thing that varies, primary arm
(half_lld imputation, environmental covariates), mean RMSE in log10 units per element
then averaged over the 39 primary-core elements:

| Protocol | K | Mean RMSE | Median | Fold-mean range |
|---|---|---|---|---|
| random_k3 | 3 | 0.1338 | 0.1248 | 0.1337 – 0.1339 |
| environmental_k3 | 3 | 0.1402 | 0.1334 | 0.1338 – 0.1446 |
| spatial_k3 | 3 | 0.1496 | 0.1409 | 0.1422 – 0.1582 |
| **loso** | 3 | **0.2677** | 0.2229 | 0.2085 – 0.3089 |
| random_k10 | 10 | 0.1311 | 0.1225 | 0.1274 – 0.1346 |

LOSO error is twice random *k*-fold error. The fold-mean ranges do not overlap:
0.2085 – 0.3089 against 0.1337 – 0.1339. Paired over the same 39 elements the
difference is +0.1339 log10 units, 95% bootstrap CI [+0.0957, +0.1853], and **all 39
of 39 elements are worse under LOSO**. The refutation condition is not met from any
direction.

The ordering of the three blocked protocols is itself a result: random < environmental
< spatial < LOSO. Spatial blocking at a measured 1,425 km range recovers about 12% of
the LOSO penalty; environmental blocking about 5%. Neither is a substitute for holding
out a survey.

---

## H3 — the penalty scales with detection-limit disparity. REFUTED.

*Refutation condition: the bootstrap interval for the slope on log Δ contains zero.*

Single computable fold pair, NGSA ↔ NASGL, 33 elements, 2,000 bootstrap resamples
(`task3_slopes.csv`):

| Direction | Elements | Slope | 95% CI | Contains zero |
|---|---|---|---|---|
| NASGL → NGSA | 33 | +0.438 | [−0.086, +0.778] | yes |
| NGSA → NASGL | 33 | +0.470 | [−0.052, +0.801] | yes |

Both slopes are positive and both intervals include zero. **Refuted as pre-registered.**
The sign is in the predicted direction in both directions, which is worth reporting as
a direction the data is consistent with, but it is not a supported claim and is not
written up as one. With n_pairs = 1, elements inside the pair share laboratory,
digestion and batch structure, so the effective sample size is far below 33 and no
generality over laboratories follows.

---

## H4 — the effect survives geography. HOLDS.

*Refutation condition: route accuracy does not exceed the best-single-element null.*

Tellus controlled arm, identical physical samples analysed by two routes, so continent,
site, custodian and sampling design are held fixed by construction
(`task4_route_identification.csv`):

| Fold | Balanced accuracy | Best-single-element null | Beats null |
|---|---|---|---|
| 0 | 0.99956 | 0.97830 | yes |
| 1 | 0.99911 | 0.97121 | yes |
| 2 | 0.99867 | 0.97741 | yes |
| 3 | 0.99911 | 0.97030 | yes |
| 4 | 0.99867 | 0.97385 | yes |

Five of five folds. The pre-registered partial observation — paired censoring
indicators disagreeing on 36 of 47 elements above 5%, median 18.45% — stands, and the
classifier now confirms it. This is the cleanest result in the paper: the only thing
differing between the two classes is the analytical route.

---

## H5 — censoring shift is separable from geochemical affinity. UNTESTED AS REGISTERED.

*Registered test: a censoring balance and an affinity-only balance compared across
protocols; refuted if the two move together.* Neither balance was computed, so H5 was
not tested as registered.

**Correction, 9 October 2026.** An earlier version of this record scored H5 as refuted
on the imputation contrast. That contrast is a different test, specified on
27 September 2026 before any Task 1 run, and it answers a narrower question. It is
reported here in its own right.

**Imputation contrast.** The LOSO-minus-random gap at matched K = 3 is flat across the
three arms: environmental covariates +0.1339 (`half_lld`), +0.1248 (`median_train`),
+0.1315 (`nan`); no covariates +0.1338, +0.1242, +0.1310; coordinates +0.1546, +0.1516,
+0.1503 log10 units, with 39/39 elements worse in every cell. All three arms leave the
censoring pattern available to the estimator (`nan` as missingness, `median_train` as a
spike of identical values). The flat gap therefore shows that the penalty does not
depend on the **value** written into censored cells. It does not show that the penalty
is independent of censoring, because the fold-internal mask regression that would
remove the pattern was not executed.

**Addendum, 9 October 2026.** The mask regression was then characterised on simulated
data, as specified, and failed its gate (survey balanced accuracy 0.589 after
residualisation against 0.315 without censoring). Common-limit re-censoring closed the
channel in simulation (0.353) but is computable for only 2 of 39 core elements, because
GEMAS publishes few limits. The censoring pattern cannot be removed with the released
data, and its share of the penalty remains open.

## What the paper can and cannot claim after this

**Supported.**

1. Censoring patterns identify analytical provenance, decisively and under every
   blocking protocol, and the identification is not geographic — it survives a
   stratified permutation at *p* = 0.000999 and survives the Tellus design, where
   geography is held fixed by construction.
2. Cross-survey generalization in Task 1 is twice as hard as random *k*-fold, on every
   one of 39 elements, and neither spatial blocking at the measured autocorrelation
   range nor environmental blocking comes close to reproducing that penalty. Reported
   random-split numbers for this kind of task are optimistic by about a factor of two.

**Not supported, and this is the central negative result.**

3. **The Task 1 cross-survey penalty does not depend on the censored value.**
   Closing that channel leaves the penalty intact. Censoring carries provenance (1),
   and cross-region validation is much harder than random validation (2), but these are
   two findings rather than one causal chain, and the paper must not present the second
   as a consequence of the first. The penalty is a distributional shift in the
   concentrations themselves, which the design was built to detect and did.
4. No dependence of the penalty on detection-limit disparity (H3), on a single fold
   pair with no generality over laboratories.
5. On GEMAS – NASGL, nothing beyond a single element's censoring indicator.

**Consequence for the framing.** The one-sentence claim that censoring-driven leakage
explains inflated cross-region generalization is not what was measured and will not be
written. What was measured is narrower and still worth a benchmark paper: censoring is
a provenance fingerprint, cross-survey evaluation is twice as punishing as random
splits, and the two facts are independent. The abstract and introduction must be
rewritten to that, and the earlier causal framing withdrawn.

---

## Provenance

Every number above is read from the released result tables, not transcribed:
`task1_*.csv` (63 runs, 14,742 element-fold rows), `task2_*.csv`, `task2p_pairwise.csv`,
`task3_slopes.csv`, `task4_route_identification.csv`,
`control_permutation_pooled.csv`. Per-element aggregates, paired differences and
bootstrap intervals in H2 and H5 were computed from those files with seed 20260925;
the paired-difference bootstrap uses 10,000 resamples in H2 and 4,000 in H5.

No run was repeated after its output was inspected, and no acceptance criterion was
changed after a result was seen.
