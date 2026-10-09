"""The five tasks of methodology 3.1, plus Task 2p and Task 4b.

One design decision is made explicit here rather than buried, because it changes what
the paper can claim:

  COVARIATES DEFAULT TO ENVIRONMENTAL, NOT COORDINATES. Feeding raw latitude and
  longitude into Task 1 makes survey identity trivially recoverable -- the three
  surveys occupy disjoint coordinate boxes -- so a coordinate-fed model under LOSO is
  extrapolating in a region it has never seen, and its failure would say nothing about
  censoring. Covariates are therefore Koppen group and GLiM lithology by default,
  which exist on every continent and are comparable across them. `coords` is available
  for a sensitivity run and is labelled as such.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import SURVEYS, TASK3_PAIR
from .data import (load_observational, core_elements, environmental_covariates,
                   load_tellus, delta_table)
from .folds import folds, task2p_splits, require_task2_eligible
from .metrics import regression_metrics, balanced_accuracy, per_class_recall
from .model import (regressors, classifiers, fit_nested, CensoredLogTransform,
                    categorical_design)
from .nulls import all_nulls
from .maskreg import residualize_on_mask


def _design(conc, lod, elements, target, strategy, train_idx, mask=None):
    """Predictor matrix for Task 1: every core element except the target.

    strategy "maskreg" = median_train imputation, then fold-internal regression of
    every predictor on the predictors' censoring mask (geoshift.maskreg).
    """
    feat = [e for e in elements if e != target]
    X = conc[feat].to_numpy(float)
    D = lod[feat].to_numpy(float)
    base = "median_train" if strategy == "maskreg" else strategy
    tr = CensoredLogTransform(strategy=base, lod=D).fit(X[train_idx])
    Xt = tr.transform(X)
    if strategy == "maskreg":
        if mask is None:
            raise ValueError("maskreg needs the censoring mask")
        Xt = residualize_on_mask(Xt, mask[feat].to_numpy(float), train_idx)
    return Xt, feat


def task1(protocol, elements=None, model="hist_gbr", imputation="half_lld",
          covariates="environmental"):
    """Cross-survey concentration prediction: hold out one element, predict it.

    Rows where the TARGET is censored are dropped from scoring: there is no observed
    value to score against, and substituting half the limit and then scoring against
    that substitution would be scoring the substitution rather than the prediction.
    The count dropped is reported per fold per element, because it varies with the
    protocol and is itself informative.
    """
    els = elements or core_elements()
    meta, conc, mask, lod = load_observational(els)
    cov = environmental_covariates(meta)
    pipe_grid = regressors()[model]
    rows = []
    for fid, train_idx, test_idx in folds(protocol, meta):
        if covariates == "environmental":
            Z, znames = categorical_design(cov, train_idx)
        elif covariates == "coords":
            Z = meta[["latitude", "longitude"]].to_numpy(float)
            znames = ["latitude", "longitude"]
        elif covariates == "none":
            Z, znames = np.zeros((len(meta), 0)), []
        else:
            raise ValueError(f"unknown covariates {covariates!r}")
        for target in els:
            Xe, feat = _design(conc, lod, els, target, imputation, train_idx, mask)
            X = np.hstack([Xe, Z]) if Z.shape[1] else Xe
            y_raw = conc[target].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log10(np.where(y_raw > 0, y_raw, np.nan))
            obs = np.isfinite(y)
            tr = train_idx[obs[train_idx]]
            te = test_idx[obs[test_idx]]
            n_drop_te = int(len(test_idx) - len(te))
            if len(tr) < 30 or len(te) < 10:
                rows.append(dict(task="1", protocol=protocol, fold=fid,
                                 element=target, model=model,
                                 imputation=imputation, covariates=covariates,
                                 n_train=len(tr), n_test=len(te),
                                 n_test_censored_dropped=n_drop_te,
                                 status="skipped_insufficient_observed"))
                continue
            pipe, grid = regressors()[model]
            est, best, inner = fit_nested(pipe, grid, X[tr], y[tr], kind="reg")
            pred = est.predict(X[te])
            m = regression_metrics(y[te], pred)
            row = dict(task="1", protocol=protocol, fold=fid, element=target,
                       model=model, imputation=imputation, covariates=covariates,
                       n_train=len(tr), n_test=len(te),
                       n_test_censored_dropped=n_drop_te,
                       censoring_rate_train=float(mask[target].to_numpy()[tr].mean()),
                       censoring_rate_test=float(mask[target].to_numpy()[te].mean()),
                       inner_cv_score=inner, best_params=str(best),
                       status="ok", **m)
            # per-survey breakdown inside the test fold (CY-Bench convention)
            sv = meta["survey"].to_numpy()[te]
            for s in SURVEYS:
                sel = sv == s
                if sel.sum() >= 10:
                    ms = regression_metrics(y[te][sel], pred[sel])
                    row[f"rmse_log10_{s}"] = ms["rmse_log10"]
                    row[f"r2_log10_{s}"] = ms["r2_log10"]
                    row[f"n_{s}"] = ms["n"]
            rows.append(row)
    return rows


def task2(protocol, elements=None, model="random_forest"):
    """Provenance identification: survey of origin from the censoring mask alone.

    Refuses to run on a protocol whose folds do not all carry three surveys at
    n >= 30. LOSO is excluded by construction; environmental_k10 and predictor_kmeans
    are excluded on measured counts.
    """
    els = elements or core_elements()
    meta, conc, mask, lod = load_observational(els)
    require_task2_eligible(protocol, meta)
    M = mask.to_numpy(np.int8)
    y = meta["survey"].to_numpy()
    rows = []
    for fid, train_idx, test_idx in folds(protocol, meta):
        pipe, grid = classifiers()[model]
        est, best, inner = fit_nested(pipe, grid, M[train_idx], y[train_idx],
                                      kind="clf")
        pred = est.predict(M[test_idx])
        ba = balanced_accuracy(y[test_idx], pred)
        nl = all_nulls(M[train_idx], y[train_idx], M[test_idx], y[test_idx], els)
        rows.append(dict(
            task="2", protocol=protocol, fold=fid, model=model,
            n_train=len(train_idx), n_test=len(test_idx),
            n_classes_test=int(len(np.unique(y[test_idx]))),
            chance=1.0 / 3.0,
            balanced_accuracy=ba, inner_cv_score=inner, best_params=str(best),
            beats_binding_null=(bool(ba > nl["binding_null_value"])
                                if np.isfinite(nl["binding_null_value"])
                                and ba is not None and np.isfinite(ba) else None),
            margin_over_binding_null=(float(ba - nl["binding_null_value"])
                                      if np.isfinite(nl["binding_null_value"])
                                      and ba is not None and np.isfinite(ba)
                                      else np.nan),
            **{f"recall_{k}": v for k, v in
               per_class_recall(y[test_idx], pred).items()},
            **nl))
    return rows


def task2p(elements=None, model="random_forest"):
    """Pairwise provenance under site-level holdout. Two classes, chance 1/2.

    This is the provenance result reported where a genuinely held-out split is
    required. It removes the single-class defect; it does NOT add a second source of
    identification, because the two surveys still differ in continent as well as in
    laboratory. That limitation is the paper's, not the harness's, and is stated in
    methodology 5.
    """
    els = elements or core_elements()
    meta, conc, mask, lod = load_observational(els)
    M = mask.to_numpy(np.int8)
    y = meta["survey"].to_numpy()
    rows = []
    for pair, train_idx, test_idx in task2p_splits(meta):
        pipe, grid = classifiers()[model]
        est, best, inner = fit_nested(pipe, grid, M[train_idx], y[train_idx],
                                      kind="clf")
        pred = est.predict(M[test_idx])
        ba = balanced_accuracy(y[test_idx], pred)
        nl = all_nulls(M[train_idx], y[train_idx], M[test_idx], y[test_idx], els)
        rows.append(dict(
            task="2p", pair=pair, model=model,
            n_train=len(train_idx), n_test=len(test_idx), chance=0.5,
            balanced_accuracy=ba, inner_cv_score=inner, best_params=str(best),
            beats_binding_null=(bool(ba > nl["binding_null_value"])
                                if np.isfinite(nl["binding_null_value"])
                                and ba is not None and np.isfinite(ba) else None),
            margin_over_binding_null=(float(ba - nl["binding_null_value"])
                                      if np.isfinite(nl["binding_null_value"])
                                      and ba is not None and np.isfinite(ba)
                                      else np.nan),
            **{f"recall_{k}": v for k, v in
               per_class_recall(y[test_idx], pred).items()},
            **nl))
    return rows


def task3(elements=None, model="hist_gbr", imputation="half_lld",
          n_boot=2000, seed=None):
    """Censoring-severity response: Task 1 performance against detection-limit
    disparity, on NGSA<->NASGL only.

    ONE fold pair, 34 element-level points. Two restrictions follow and are enforced
    here rather than left to the write-up:
      - Elements share a laboratory, digestion and batch structure within the pair, so
        they are not independent draws. The slope is bootstrapped over ELEMENTS and
        reported with that dependence named as unmodelled.
      - Across pairs n = 1. No generality over laboratories follows, and the returned
        rows carry `n_pairs=1` so a later aggregation cannot forget it.
    """
    from .config import TASK3_PAIR as PAIR
    els = elements or core_elements()
    meta, conc, mask, lod = load_observational(els)
    a, b = PAIR
    sv = meta["survey"].to_numpy()
    idx_a, idx_b = np.flatnonzero(sv == a), np.flatnonzero(sv == b)
    dt = delta_table()
    col = None
    for c in dt.columns:
        if {a.lower(), b.lower()} <= set(str(c).lower().replace("_over_", "_").split("_")):
            col = c
            break
    if col is None:
        cands = [c for c in dt.columns
                 if a.lower() in str(c).lower() and b.lower() in str(c).lower()]
        if not cands:
            raise ValueError(
                f"no disparity column for {a}<->{b} in lod-disparity-delta.csv; "
                f"columns are {list(dt.columns)}")
        col = cands[0]
    dt = dt.set_index(dt.columns[0])
    rows = []
    for direction, train_idx, test_idx in ((f"{a}->{b}", idx_a, idx_b),
                                           (f"{b}->{a}", idx_b, idx_a)):
        for target in els:
            if target not in dt.index:
                continue
            ratio = pd.to_numeric(pd.Series([dt.loc[target, col]]),
                                  errors="coerce").iloc[0]
            if not np.isfinite(ratio) or ratio <= 0:
                continue
            delta = float(max(ratio, 1.0 / ratio))       # fold-symmetric
            Xe, feat = _design(conc, lod, els, target, imputation, train_idx)
            y_raw = conc[target].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log10(np.where(y_raw > 0, y_raw, np.nan))
            obs = np.isfinite(y)
            tr, te = train_idx[obs[train_idx]], test_idx[obs[test_idx]]
            if len(tr) < 30 or len(te) < 10:
                continue
            pipe, grid = regressors()[model]
            est, best, inner = fit_nested(pipe, grid, Xe[tr], y[tr], kind="reg")
            m = regression_metrics(y[te], est.predict(Xe[te]))
            rows.append(dict(task="3", pair=f"{a}_{b}", direction=direction,
                             element=target, delta=delta, log10_delta=float(np.log10(delta)),
                             model=model, imputation=imputation,
                             n_train=len(tr), n_test=len(te), n_pairs=1, **m))
    df = pd.DataFrame(rows)
    slopes = []
    if len(df):
        rng = np.random.default_rng(seed if seed is not None else 20260925)
        for direction, g in df.groupby("direction"):
            g = g[np.isfinite(g.nrmse_log10) & np.isfinite(g.log10_delta)]
            if len(g) < 5:
                continue
            x, yv = g.log10_delta.to_numpy(), g.nrmse_log10.to_numpy()
            beta = float(np.polyfit(x, yv, 1)[0])
            bs = np.empty(n_boot)
            n = len(g)
            for i in range(n_boot):
                s = rng.integers(0, n, n)          # bootstrap over ELEMENTS
                bs[i] = np.polyfit(x[s], yv[s], 1)[0]
            slopes.append(dict(
                task="3", direction=direction, n_elements=n, slope=beta,
                ci_lo=float(np.percentile(bs, 2.5)),
                ci_hi=float(np.percentile(bs, 97.5)),
                n_boot=n_boot, n_pairs=1,
                caveat=("elements within a pair share laboratory, digestion and "
                        "batch structure; residual dependence is not modelled, and "
                        "n_pairs = 1 so no generality over laboratories follows")))
    return rows, slopes


def task4(extended=False, model="random_forest"):
    """Route identification under fixed geography, on Tellus. Chance 1/2.

    The controlled arm: same physical sample, two analytical routes, so site, sampler
    and custodian are held constant and only the route varies. Each sample appears
    once per route, and the split is over SAMPLES, so no sample's two route rows land
    on opposite sides.

    `extended` (Task 4b) adds the elements excluded from the three-survey core, which
    is where the largest measured route effects sit (Hf 52.3%, Ge 28.7%, Zr 26.2%).
    Reported separately and never pooled with the 39-element version.
    """
    FX, FI, els, rows_ok, prov = load_tellus()
    if not extended:
        keep = prov[prov.both_accredited].element.tolist()
        els = [e for e in els if e in keep]
    ok = rows_ok.to_numpy()
    A = FX.loc[ok, els].to_numpy(np.int8)
    B = FI.loc[ok, els].to_numpy(np.int8)
    n = A.shape[0]
    X = np.vstack([A, B])
    y = np.array(["XRFS"] * n + ["ICPar"] * n)
    sample = np.concatenate([np.arange(n), np.arange(n)])
    rng = np.random.default_rng(20260925)
    order = rng.permutation(n)
    k = 5
    rows = []
    for fold in range(k):
        te_s = set(order[fold::k].tolist())
        te = np.flatnonzero(np.isin(sample, list(te_s)))
        tr = np.flatnonzero(~np.isin(sample, list(te_s)))
        pipe, grid = classifiers()[model]
        est, best, inner = fit_nested(pipe, grid, X[tr], y[tr], kind="clf")
        pred = est.predict(X[te])
        ba = balanced_accuracy(y[te], pred)
        nl = all_nulls(X[tr], y[tr], X[te], y[te], els)
        rows.append(dict(task="4b" if extended else "4", fold=fold, model=model,
                         n_elements=len(els), n_samples=n,
                         n_train=len(tr), n_test=len(te), chance=0.5,
                         balanced_accuracy=ba, inner_cv_score=inner,
                         best_params=str(best),
                         beats_binding_null=(bool(ba > nl["binding_null_value"])
                                             if np.isfinite(nl["binding_null_value"])
                                             and ba is not None and np.isfinite(ba)
                                             else None),
                         split_unit="sample (both route rows move together)",
                         **nl))
    return rows
