"""The three confound controls of methodology 3.4.

1. Stratified label permutation on Koppen x GLiM strata. Geographic strata would give
   an identity permutation, because the three surveys are disjoint in space, so the
   control would have no power while appearing to run. Environmental strata are used
   and the REALISED POWER IS REPORTED BEFORE the p-value is interpreted -- a p-value
   from a degenerate permutation is not a weak result, it is not a result.

   The strata are joined on survey + coordinates, NOT on row order. An earlier draft
   of this file reindexed the strata frame to range(len(meta)), which silently shifted
   every stratum after the first coordinate-free sample onto the wrong row: the strata
   file has 8,280 rows against the data's 8,285. That would have corrupted this
   control while producing a plausible number.

2. Fold-internal mask regression, after Snoek et al. (2019), who show the
   full-dataset variant drives performance below chance. Only the fold-internal
   version is unbiased. Snoek treats the confound as external to the feature matrix;
   here the mask is internal -- it indexes the same measurements it confounds -- so
   the adaptation is characterised rather than assumed equivalent.

3. Paired limits of agreement on the censoring INDICATORS, not concentrations. XRFS
   is near-total and aqua regia a partial digest, so a concentration bias between them
   is expected by design and is not disagreement in Bland and Altman's sense.
"""
from __future__ import annotations
from math import lgamma
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import SEED
from .data import load_observational, core_elements, environmental_covariates, load_tellus
from .metrics import balanced_accuracy
from .model import classifiers, fit_nested


def build_strata(meta, definition="climate_x_lithology"):
    """Stratum label per sample, or NaN where none resolves. Joined on coordinates."""
    cov = environmental_covariates(meta)
    keys = {"climate_x_lithology": ["kg_group", "glim_class"],
            "climate_only": ["kg_group"],
            "lithology_only": ["glim_class"]}[definition]

    # Built by plain string concatenation, not DataFrame.agg("|".join, axis=1). That
    # form hands pandas each row as a Series and relies on every element already being
    # a str; a single non-string value raises inside join, which is what killed the
    # first attempt. Concatenation on a nullable string dtype is total: every value
    # either joins or is explicitly marked missing.
    parts = []
    missing = pd.Series(False, index=cov.index)
    for k in keys:
        s = cov[k].astype("string")
        missing |= s.isna()
        parts.append(s.fillna(""))
    lab = parts[0]
    for p in parts[1:]:
        lab = lab + "|" + p
    lab = lab.astype(object)
    lab[missing.to_numpy()] = np.nan
    return lab


def permutation_power(strata, y):
    """Realised power of the stratified permutation, computed before any test.

    log_permutations is the log of the number of distinct within-stratum label
    arrangements: sum over multi-survey strata of log(n! / prod(n_c!)). If it is zero
    the permutation is the identity and the control bounds nothing.
    """
    df = pd.DataFrame({"s": strata, "y": y}).dropna()
    tab = pd.crosstab(df.s, df.y)
    multi = tab[(tab > 0).sum(axis=1) >= 2]
    ent = 0.0
    for _, r in multi.iterrows():
        n = int(r.sum())
        ent += lgamma(n + 1) - sum(lgamma(int(c) + 1) for c in r if c > 0)
    return dict(strata=int(len(tab)), multi_survey_strata=int(len(multi)),
                samples_in_multi=int(multi.to_numpy().sum()),
                pct_samples_in_multi=(round(100 * multi.to_numpy().sum() / len(df), 2)
                                      if len(df) else 0.0),
                log_permutations=round(ent, 1),
                surveys_represented=int(multi.gt(0).any(axis=0).sum()) if len(multi) else 0,
                degenerate=bool(len(multi) == 0 or ent == 0.0)), tab


def stratified_permutation_test(n_perm=1000, definition="climate_x_lithology",
                               model="random_forest", elements=None,
                               train_frac=0.7, seed=SEED, perm_seed=None):
    """Permute survey labels WITHIN strata and re-run the Task 2 classifier.

    An observed accuracy that survives permutation is instrumental; one that does not
    is geographic or the test is underpowered -- and those two are NOT the same
    conclusion, so the power figures above are returned alongside and must be read
    first.

    Cost warning: this fits the classifier n_perm + 1 times. At n_perm = 1000 on the
    full mask that is hours, not minutes. Reduce n_perm for a smoke run and say so in
    the write-up; do not present a reduced-permutation p-value as the final one.
    """
    els = elements or core_elements()
    meta, conc, mask, lod = load_observational(els)
    M = mask.to_numpy(np.int8)
    y = meta["survey"].to_numpy()
    strata = build_strata(meta, definition).to_numpy()
    keep = pd.notna(strata)
    M, y, strata = M[keep], y[keep], strata[keep]

    power, tab = permutation_power(strata, y)
    power["definition"] = definition
    power["n_used"] = int(keep.sum())
    power["n_excluded_no_stratum"] = int((~keep).sum())
    if power["degenerate"]:
        power["verdict"] = ("IDENTITY PERMUTATION - the control has no power and the "
                            "confound is unbounded. No p-value is reported.")
        return power, None

    # TWO separate RNGs, deliberately. The train/test split is drawn from `seed` only,
    # so it is byte-identical in every part of a split run and the observed accuracy is
    # the same number each time. The permutation draws come from `perm_seed`, so each
    # part explores different label arrangements. Sharing one RNG would change the
    # split per part, the observed accuracy would differ, and the parts could not be
    # pooled into one null distribution.
    split_rng = np.random.default_rng(seed)
    idx = split_rng.permutation(len(y))
    cut = int(train_frac * len(y))
    tr, te = idx[:cut], idx[cut:]
    pipe, grid = classifiers()[model]
    est, best, _ = fit_nested(pipe, grid, M[tr], y[tr], kind="clf")
    observed = balanced_accuracy(y[te], est.predict(M[te]))

    rng = np.random.default_rng(seed if perm_seed is None else perm_seed)
    order = np.arange(len(y))
    by_stratum = {s: order[strata == s] for s in np.unique(strata)}
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = y.copy()
        for s, ix in by_stratum.items():
            yp[ix] = rng.permutation(y[ix])
        est_b, _, _ = fit_nested(pipe, grid, M[tr], yp[tr], kind="clf")
        null[b] = balanced_accuracy(yp[te], est_b.predict(M[te]))
    finite = null[np.isfinite(null)]
    p = float((1 + (finite >= observed).sum()) / (1 + finite.size))
    power.update(model=model, n_perm=n_perm, n_perm_finite=int(finite.size),
                 observed_balanced_accuracy=float(observed),
                 permutation_mean=float(finite.mean()),
                 permutation_q95=float(np.quantile(finite, 0.95)),
                 p_value=p, best_params=str(best),
                 verdict=("survives stratified permutation - the signal is "
                          "instrumental, not geographic"
                          if p < 0.05 else
                          "does not survive - geographic, or the test is "
                          "underpowered; these are different conclusions and the "
                          "power figures above decide which"))
    return power, finite


def fold_internal_mask_regression(X, M, train_idx):
    """Regress the mask out of the predictors, fitting WITHIN the training fold.

    Returns residualised X for ALL rows, with coefficients estimated on train only.
    Fitting on the full dataset is what Snoek et al. show drives performance below
    chance, so the train_idx argument is required rather than optional.
    """
    X = np.asarray(X, float)
    M = np.asarray(M, float)
    finite = np.isfinite(X)
    med = np.nanmedian(np.where(finite, X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(finite, X, med)
    lr = LinearRegression().fit(M[train_idx], Xf[train_idx])
    return Xf - lr.predict(M)


def paired_limits_of_agreement():
    """Bland and Altman on the paired censoring indicators, per element.

    Applied to indicators and not concentrations: the two routes measure different
    quantities by design, so a concentration bias is expected and is not disagreement.
    Elements whose limit provenance is non-accredited on either route are flagged and
    excluded from headline summaries rather than dropped silently.
    """
    FX, FI, els, rows_ok, prov = load_tellus()
    ok = rows_ok.to_numpy()
    pmap = prov.set_index("element")
    rows = []
    for e in els:
        a = FX[e].to_numpy()[ok].astype(float)
        b = FI[e].to_numpy()[ok].astype(float)
        d = a - b
        sd = float(d.std(ddof=1))
        rows.append(dict(
            element=e, n=int(ok.sum()),
            xrfs_censoring_rate=float(a.mean()),
            icpar_censoring_rate=float(b.mean()),
            bias=float(d.mean()), sd=sd,
            loa_lower=float(d.mean() - 1.96 * sd),
            loa_upper=float(d.mean() + 1.96 * sd),
            disagree_rate=float((a != b).mean()),
            icpar_censors_more=bool(b.mean() > a.mean()),
            xrfs_limit=pmap.loc[e, "xrfs_limit"],
            icpar_limit=pmap.loc[e, "icpar_limit"],
            both_accredited=bool(pmap.loc[e, "both_accredited"]),
            in_headline=bool(pmap.loc[e, "both_accredited"])))
    return rows
