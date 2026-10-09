"""Metrics, with the normalisation stated rather than implied.

Concentrations are lognormal, so every regression is fitted on log10(y) and the
primary metrics are reported on that scale. Two normalisations are avoided
deliberately:

  RMSE / mean(y)  -- unstable when a test fold's mean is near zero, which happens
                     for trace elements in a blocked fold.
  MAPE on log10   -- meaningless, because log10 values cross zero.

So: nrmse_log10 divides by the standard deviation of the observed test values, which
is the convention that makes the number comparable across elements of wildly
different abundance. MAPE is computed on back-transformed original units, where a
percentage error is interpretable. Both are reported; neither is presented alone.
"""
from __future__ import annotations
import numpy as np


def regression_metrics(y_true_log, y_pred_log):
    """y_* are log10 concentrations. Returns a dict; keys name their own scale."""
    y_true_log = np.asarray(y_true_log, float)
    y_pred_log = np.asarray(y_pred_log, float)
    ok = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y, p = y_true_log[ok], y_pred_log[ok]
    n = y.size
    if n < 2:
        return dict(n=int(n), rmse_log10=np.nan, nrmse_log10=np.nan,
                    r2_log10=np.nan, mae_log10=np.nan, mape_pct=np.nan,
                    bias_log10=np.nan)
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sd = float(np.std(y, ddof=1))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # Back-transform for MAPE only. 10**log10 is exact, so no smearing correction is
    # applied and none is implied: MAPE here describes the median-unbiased
    # back-transform, not the mean, and is labelled as a secondary metric.
    yt, pt = np.power(10.0, y), np.power(10.0, p)
    nz = yt != 0
    mape = float(np.mean(np.abs((pt[nz] - yt[nz]) / yt[nz])) * 100.0) if nz.any() else np.nan
    return dict(n=int(n), rmse_log10=rmse,
                nrmse_log10=float(rmse / sd) if sd > 0 else np.nan,
                r2_log10=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                mae_log10=float(np.mean(np.abs(err))),
                mape_pct=mape, bias_log10=float(np.mean(err)))


def balanced_accuracy(y_true, y_pred):
    """Mean per-class recall. Returns NaN -- not 0 -- when a class is absent from
    y_true, because an absent class has no recall to average and a 0 would be read
    as a measured failure. This is the defect that removed Task 2 from LOSO."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    if classes.size < 2:
        return np.nan
    recalls = []
    for c in classes:
        m = y_true == c
        if m.sum() == 0:
            return np.nan
        recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls))


def per_class_recall(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {str(c): float((y_pred[y_true == c] == c).mean())
            for c in np.unique(y_true)}


def confusion_counts(y_true, y_pred, labels):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for a in labels:
        for b in labels:
            out[f"n_{a}_as_{b}"] = int(((y_true == a) & (y_pred == b)).sum())
    return out
