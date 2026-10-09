"""Fold-internal censoring-mask regression (methodology Sect. 2.10 / §3.4.2).

The only treatment in the design that removes the censoring *pattern* rather than
the censored *value*. Following Snoek et al. (2019), each predictor is regressed on
the censoring mask of all predictors, the coefficients are fitted on the training
fold only, and the same coefficients residualise the test fold. Fitting once on the
full data is the variant Snoek et al. show to be biased, and is not offered.

Base imputation is `median_train`, which already removes the survey-specific value;
residualisation then removes what the mask still explains linearly. Whether that
closes the channel for a tree model is an empirical question, answered on simulated
data by `check_maskreg_simulation.py` before any real run.
"""
from __future__ import annotations
import numpy as np


def residualize_on_mask(X, M, train_idx):
    """Return X with its linear dependence on the mask M removed.

    X          n x p predictors, no NaN (already imputed)
    M          n x q binary censoring mask
    train_idx  rows the regression is fitted on; applied to all rows
    """
    X = np.asarray(X, float)
    A = np.hstack([np.ones((X.shape[0], 1)), np.asarray(M, float)])
    coef, *_ = np.linalg.lstsq(A[train_idx], X[train_idx], rcond=None)
    return X - A @ coef
