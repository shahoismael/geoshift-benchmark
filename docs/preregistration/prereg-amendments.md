# Pre-registration amendment log — GeoShift

Pre-registration deposited 26 September 2026.
Deposited file SHA-256: `7907BC7CE84A096188A9E6816E0BA5E970F80C09CDCB7E8A3BF002A43317ACED`

**No task has been fitted on real data at the time of any amendment below.** Each was
forced by a check on the design or on the fold files, not by a result. All three narrow
what the benchmark can claim. The superseded fold files are retained, not deleted.

---

## Amendment (a) — Task 2 withdrawn from leave-one-survey-out

**Date** 26 September 2026
**Affects** H2; methodology §3.1, §3.2, §3.5

**What was pre-registered.** Task 2 (predict survey of origin from the censoring mask)
estimated under all protocols including LOSO, chance 1/3.

**What is wrong with it.** Under LOSO the held-out fold contains exactly one survey.
The test set therefore has one class and the other two are absent, so balanced accuracy
has no value — it returns 0 on a three-class scorer. This is not repairable by
one-vs-rest or by a pairwise recoding: the defect is the absent classes in the *test
set*, not the number of classes in the scorer.

**How it was found.** A synthetic fixture built with the real schema and planted
per-survey detection limits — so survey identity is recoverable by construction —
returned balanced accuracy **0.000 on every LOSO fold**. A fixture with planted ground
truth that scores zero indicts the harness or the design, not the hypothesis.

**The amendment.** Task 2 is estimated under random *k*-fold, spatial blocking and
environmental blocking, whose folds each carry all three surveys. It is **not**
estimated under LOSO and no number is reported there. A new task is added to supply a
genuinely held-out provenance split:

> **Task 2p, pairwise provenance under site-level holdout.** For each survey pair,
> sites are split 80/20 within each of the two surveys; the classifier trains on both
> surveys' training sites and tests on both surveys' held-out sites. Two classes in
> the test set, chance 1/2, no site on both sides. Three pairs.

**What this costs the paper.** Task 2p holds out sites, not surveys, so its two classes
still differ in continent as well as in laboratory — it removes the single-class defect
without adding a second source of identification. The claim that provenance is
recoverable from the mask under a genuinely held-out split therefore rests on Task 2p
together with Task 4, where geography is fixed, and not on Task 2 alone.

---

## Amendment (b) — NNDM-geographic confined to within-survey use

**Date** 26 September 2026
**Affects** H4; methodology §3.3

**What was pre-registered.** NNDM LOO-CV (Milà et al., 2022) as one of the cross-survey
protocols, on geographic distance.

**What is wrong with it.** NNDM deletes training points until the training-to-test
nearest-neighbour distance distribution matches the prediction-to-training distribution
for the target area. Measured on the released coordinates, the median distance from a
sample to the nearest sample of a *different* survey is:

| Survey | Median distance to nearest other-survey sample |
|---|---|
| GEMAS | 5,475 km |
| NGSA | 11,922 km |
| NASGL | 6,525 km |

These are three to four orders of magnitude beyond any plausible soil-geochemical
autocorrelation range, and beyond the largest range a variogram on these data can
return. Matching that distribution would require deleting the entire training set.

**The amendment.** NNDM-geographic is reported **within each survey only**. Its
cross-survey cell in the protocol comparison is left empty rather than filled with a
number the algorithm cannot produce. NNDM-mask (Jaccard distance on the censoring mask)
is unaffected and remains the extension the paper contributes; the measured Jaccard
target distributions (GEMAS 0.000, NGSA 0.375, NASGL 0.500) are not degenerate, and
GEMAS's zero is recorded as a metadata-availability artefact.

**What this costs the paper.** One of five protocols loses its cross-survey estimate.
The protocol grid is complete for Task 1 under four protocols, not five.

---

## Amendment (c) — spatial block side measured; environmental folds rebuilt

**Date** 26 September 2026
**Affects** H1, H3; methodology §3.3; fold files

### (c1) Spatial block side

**What was pre-registered.** `folds-spatial-block.csv` built on a 2° grid.

**What is wrong with it.** Roberts et al. (2017) require the block to be at least as
large as the range of residual spatial autocorrelation. A block smaller than that range
leaves training points inside the test block's correlation neighbourhood and the
protocol silently reverts towards random *k*-fold. That failure direction matters here:
it would understate the gap between blocked and random validation, which is the paper's
central comparison — the error would flatter the paper's own thesis. The 2° grid was a
convenience with no measurement behind it.

**The amendment.** The block side *b* is fixed by measurement, before any task is
fitted: empirical variogram per primary-core element per survey, on log concentrations
after removing a first-order coordinate trend; exponential model; practical range = 3a
(95% of sill); *b* = **90th percentile** of accepted ranges across the 39 core elements,
rounded up to the next 25 km. Sensitivity blocks at 0.5*b* and 2*b*. The 90th
percentile rather than the median so that the block covers the range for nine elements
in ten; erring large costs fold resolution, erring small costs the protocol's validity.
Measured by `measure_autocorrelation_range.py`.

**Status: RUN, 27 September 2026.** 50 of 117 element–survey variograms yield an
identifiable practical range (median 910 km, p75 1,156 km, p90 1,415 km) →
**b = 1,425 km**, sensitivities 712 km and 2,850 km.

**Two consequences that were not anticipated when this amendment was written.**

*(c1-i) The range is unmeasurable for most elements, and for NGSA almost entirely.*
62 of 117 variograms never reach a sill inside the observable window — GEMAS 10/39,
NASGL 17/39, **NGSA 35/39**. Only Fe (952 km) is identifiable for NGSA. So *b* is
measured on GEMAS and NASGL and applied to NGSA, and **spatial cross-validation cannot
be calibrated on NGSA at its sampling density** (1,315 sites over Australia, median
nearest-neighbour distance 29 km, continental extent ~3,600 km). Reported as a
limitation and as a finding about low-density national surveys, not presented as a
measurement on all three.

*(c1-ii) The measured block size caps the number of spatial folds at three, which
forced two further changes.* At b = 1,425 km the surveys hold 12, 10 and 10 blocks, and
blocks are survey-exclusive because the surveys are disjoint. Ten folds would give one
block per fold, with measured block populations of 3 to 1,016 samples — a fold holding
3 samples of a survey is not an estimate, and for Task 2 it is a fold missing a class.
A rule fixed before the block count was known — every spatial fold draws on ≥3 blocks
from every survey — gives **K_spatial = ⌊10/3⌋ = 3**, with measured per-fold
populations GEMAS 702–703, NASGL 1,613–1,628, NGSA 420–464. Therefore:

- **Matched-K comparison (new).** Spatial blocking at K = 3 against random at K = 10
  compares 67% training data against 90%, confounding the blocking axis with
  training-set size. `fold_random_k3` and `fold_environmental_k3` are added and **the
  headline protocol comparison runs at K = 3 throughout**. The K = 10 versions become a
  secondary read of the unblocked and environmental protocols only.
- **Three-fold inference limit widened (new).** The pre-registration restricted
  significance claims on three folds to LOSO. With the headline comparison at matched
  K = 3, that restriction now binds **every protocol in it**. No p-value is attached to
  any protocol difference; effect sizes with full per-fold values are reported instead.
  This narrows what H1 and H3 can assert and is recorded as such.

### (c2) Environmental folds

**What was pre-registered.** `folds-environmental.csv` built by *k*-means (K = 10) on
standardised log concentrations, described as environmental blocking after Valavi et al.
(2019).

**What is wrong with it.** Clustering on log concentrations blocks on the
**measurements**, not on environmental covariates, which is not what Valavi et al.
define. Worse for this paper specifically: concentrations carry the censoring structure
that Tasks 2 and 3 are about, so the comparator would be blocking partly on the object
of study.

**The amendment.** Environmental folds are built by assigning the 47 Köppen climate
group × GLiM lithology strata (§3.4.1) **whole** to 10 folds, balanced on stratum size,
so no stratum straddles the train/test boundary. The *k*-means column is retained as a
released sensitivity, renamed `fold_predictor_kmeans`, and named for what it is.

**Consequence for a pre-registered number.** The pre-registration records
*"environmental blocking partially recovers survey identity — only 5 of 10 clusters
contain all three surveys."* That was measured on the withdrawn *k*-means definition and
is **SUPERSEDED pending recomputation** on the strata-based folds by
`build_fold_definitions_v2.py`. It is not carried forward and not quoted anywhere until
recomputed.

**Status: RUN, 27 September 2026.** Recomputed, and the withdrawn definition is worse
than first reported: *k*-means at K = 10 has all three surveys present in 5 of 10 folds
and **0 of 10 folds with at least 30 samples of each**, with four folds containing no
NGSA sample at all. The pre-registered figure is superseded by that.

**It also turned into a reportable observation.** Clustering these three surveys in
geochemical predictor space nearly recovers survey identity with no model fitted. That
is why the variant cannot serve as an environmental comparator, and it is independent
support for the premise that the measurements carry provenance rather than only the
censoring mask. It is reported descriptively, not as a task result.

**The strata-based rebuild passes, with one exclusion.** Environmental blocking on the
47 Köppen × GLiM strata gives fold sizes 738–1,421 at K = 10, but three folds fail the
per-survey population criterion (GEMAS = 5, NGSA = 0, NGSA = 12), so **environmental
K = 10 is excluded from Task 2** on the same rule that excluded LOSO, and the matched-K
version (`fold_environmental_k3`) is the one Task 2 uses. 201 samples are unusable for
environmental blocking (unresolvable climate or lithology class).

### (c3) Strata join corrected from row order to coordinates — new, 27 September 2026

**What was wrong.** `sample-environmental-strata.csv` holds 8,280 rows against the
8,285-row sample table, having dropped the samples without coordinates. The earlier
harness draft aligned strata to samples **positionally**, which shifts every stratum
after the first dropped row onto the wrong sample — a wrong answer that runs clean and
would have silently corrupted the §3.4.1 permutation control.

**The amendment.** Strata are joined on survey + coordinates, with a one-to-one check
that refuses to proceed on a bad join. Measured: 8,280 of 8,285 matched, the 5 unmatched
being exactly the coordinate-free samples; one co-located duplicate site, which agrees
on its stratum. `controls.py` in the evaluation harness carries the same positional
defect and must be corrected before the permutation control is run.

---

## Amendment (d) — the Task 1 covariate vector is specified

**Date** 27 September 2026 (methodology v7.1)
**Affects** H1, H3; methodology §3.1, §3.2, §3.3
**Status** specified before any model was fitted; no task has been run

**What was pre-registered.** Task 1 as y(i) = f(**x**₋ⱼ(i), **z**(i); θ), with **z**(i)
named but never defined. That is a gap rather than a decision, and it is the single most
consequential free choice in the design — so leaving it to be settled at run time would
have made the pre-registration ineffective on exactly the point where it matters most.

**Why it matters.** A protocol comparison is interpretable only if the covariate set is
**equally informative under every protocol compared**. Otherwise the gap between
protocols measures the covariates and not the blocking axis. Raw coordinates fail that
test in this design, and fail it in the direction that flatters the paper:

| | Random *k*-fold | LOSO |
|---|---|---|
| Nearest training sample | 23–41 km | 5,475–11,922 km |
| Against measured range | 900–1,400 km | 900–1,400 km |
| Model regime | interpolation | extrapolation outside the hull |

A coordinate-fed model would therefore show a large random-versus-LOSO gap that is
interpolation against extrapolation and has nothing to do with censoring — and that gap
would have become the paper's headline number. Köppen group and GLiM class pass the
test: categorical, transferable, present on all three continents, and 35 of the 47
strata are multi-survey.

**The specification.**

- **z**(i) = one-hot Köppen–Geiger climate group × GLiM lithological class, categories
  learned on the training fold, an unseen category encoded as all-zeros.
- Three arms are run: `environmental` (primary), `none` (attribution — shows the gap
  survives with no covariate help), `coords` (the competing explanation, run in order to
  exclude geography by measurement rather than by assertion).
- Rows whose target element is itself censored are excluded from scoring, with the
  excluded count reported per fold per element. Scoring a half-limit substitution
  against itself scores the substitution, not the prediction.

**The identification lever is the imputation contrast, not the covariate choice.**
`half_lld` writes each survey's own detection limit into the feature matrix; `nan`
leaves it as a missingness pattern; `median_train` overwrites it with a training-fold
median. The change in the random-versus-LOSO gap between `half_lld` and `median_train`
is attributable to the censoring channel specifically, and that contrast is Task 1's
contribution to the claim.

**One thing deliberately not claimed.** `median_train` **attenuates** the channel, it
does not close it: it removes the value but not the pattern, since every censored cell
of an element receives the same number and a spike of identical values remains
detectable by a tree. A closure claim belongs only to §3.4.2, the fold-internal mask
regression.

---

## Run order

```powershell
python "D:\claude_projects\Geochemistry\01-paper2\05-prereg\measure_autocorrelation_range.py"
python "D:\claude_projects\Geochemistry\01-paper2\05-prereg\build_fold_definitions_v2.py"
```

The second script refuses to run without the first, because it will not guess a block
size.

## What is NOT amended

- Seed 20260925, K = 10, and the 39-element primary core are unchanged.
- The five hypotheses H1–H5 and their refutation conditions are unchanged in substance;
  H2's estimation protocol changes per amendment (a) and H4's per (b).
- The best-single-element null and its use as the acceptance criterion for Tasks 2, 2p,
  4 and 4b are unchanged.
- No amendment relaxes a refutation condition or widens a claim.
