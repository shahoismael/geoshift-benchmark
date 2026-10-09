[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23269524.svg)](https://doi.org/10.5281/zenodo.23269524)

# GeoShift


A benchmark for cross-survey evaluation of geochemical machine learning under detection-limit censoring.

GeoShift harmonises three continental-scale geochemical surveys into an **observational arm** (GEMAS, Europe; NGSA, Australia; NASGL, conterminous USA; 8,285 samples, 39 shared elements) and adds the Tellus survey of Ireland as a **controlled arm** (5,643 samples, each analysed by two analytical routes). Every cell carries censoring indicators, detection limits and limit provenance. Frozen fold definitions are released for random, spatially blocked, environmentally blocked and leave-one-survey-out validation.

> Sulaiman, D. A., Hassen, S. I., Abdlrazaq, A. A., and Noori, E. M.: GeoShift: a four-survey geochemical benchmark for cross-survey evaluation under detection-limit censoring, *Earth System Science Data* (in preparation).

## Contents

| Folder | Contents |
|---|---|
| `harness/` | evaluation harness (`run_geoshift.py`, package `geoshift/`), permutation pooling, simulation and feasibility checks |
| `audit/` | scripts that built the crosswalk, censoring metadata, detection-limit tables, strata, variogram ranges and folds |
| `folds/` | frozen fold definitions (seed 20260925), primary core element list, variogram ranges, NNDM distances |
| `metadata/` | element crosswalk, exclusions, detection-limit tables, environmental strata, Tellus censoring catalogue |
| `results/` | every reported result table, with one provenance sidecar per Task 1 run |
| `docs/preregistration/` | pre-registration, logged amendments and scored outcomes |

## Data

The harmonised tables are archived on Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX

**GEMAS values are not redistributed.** They are published by Geological Survey Ireland on data.gov.ie under CC BY 4.0, and `audit/verify_gemas_gsi.py` fetches and verifies them record for record. NGSA (CC BY 4.0), NASGL (US public domain) and Tellus (CC BY 4.0) are redistributed in harmonised form.

## Reproducing the results

Python 3.11. Install with `pip install -r requirements.txt`, place the harmonised tables in `data/interim/` and the Tellus workbook in `data/tellus/` (or point the variables `GEOSHIFT_INTERIM`, `GEOSHIFT_TELLUS`, `GEOSHIFT_FOLDS`, `GEOSHIFT_METADATA`, `GEOSHIFT_RESULTS` elsewhere), then:

```
python harness/run_geoshift.py --check
python harness/run_geoshift.py --task 1 --protocol loso --imputation half_lld --covariates environmental
python harness/run_geoshift.py --task 2 --protocol random_k3
python harness/run_geoshift.py --task 4
```

Task 1 options: `--protocol` random_k10, random_k3, spatial_k3, environmental_k10, environmental_k3, loso, predictor_kmeans; `--imputation` half_lld, median_train, nan, maskreg; `--covariates` environmental, none, coords. Each run writes a CSV and a provenance sidecar (seed, library versions, input hashes).

## Status of specified components

Executed: Tasks 1, 2, 2p, 3, 4 and 4b, the stratified permutation control and paired route agreement. Characterised and not run on real data: fold-internal mask regression, which failed its simulation check (`results/maskreg-simulation-check.csv`). Specified and not executed: NNDM, Task 5, area of applicability, compositional imputation. Details in `docs/preregistration/prereg-outcomes.md`.

## Licence

Code: MIT (`LICENSE`). Data and derived tables: CC BY 4.0. Cite the original surveys when using the data.

## Contact

Shaho Hassen, Salahaddin University, Erbil Ã¢â‚¬â€ shaho.hassen@su.edu.krd Ã¢â‚¬â€ ORCID 0000-0002-6403-7748
