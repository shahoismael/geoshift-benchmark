"""Estimators, fold-internal preprocessing, and nested cross-validation.

Two rules from the methodology are enforced structurally rather than by convention,
because both are easy to violate by accident and invisible afterwards:

  Fold-internal scaling (3.3). Centring, scaling and imputation are fitted on the
  TRAINING fold only. Under LOSO this matters more than usual: the held-out survey has
  its own location and scale, and a transform fitted on the pooled data would leak it
  into training. Everything here lives inside an sklearn Pipeline, so fitting on a
  training fold cannot reach the test fold.

  Nested CV (3.5). Hyperparameters are chosen by an inner split of the TRAINING fold.
  Choosing them on the outer test fold is the standard way to manufacture an
  optimistic number, and Bates et al. (2024) show even correct CV intervals
  undercover, so the inner loop is not optional.

Imputation is a treatment, not a detail. `half_lld` substitutes half the recorded
detection limit -- which writes the survey's own limit into the feature matrix and is
therefore one of the leakage channels the paper studies. `nan` leaves censored cells
missing and lets HistGradientBoosting route them natively, which leaks the mask as a
missingness pattern instead. Both are run; neither is presented as neutral.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import SEED

INNER_K = 3          # inner folds for hyperparameter selection
N_JOBS = -1


class CensoredLogTransform(BaseEstimator, TransformerMixin):
    """log10 with censored cells handled by an explicitly named strategy.

    X is expected as a float array of concentrations with NaN where censored, paired
    with `lod` of the same shape carrying the detection limit on censored cells. The
    lod array is passed at construction because sklearn's fit/transform signature has
    no room for it; it is sliced to the fold by the caller.
    """

    def __init__(self, strategy="half_lld", lod=None):
        self.strategy = strategy
        self.lod = lod

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log10(np.where(X > 0, X, np.nan))
        self.train_median_ = np.nanmedian(L, axis=0)
        self.train_median_ = np.where(np.isfinite(self.train_median_),
                                      self.train_median_, 0.0)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log10(np.where(X > 0, X, np.nan))
        if self.strategy == "nan":
            return L
        if self.strategy == "median_train":
            return np.where(np.isfinite(L), L, self.train_median_)
        if self.strategy == "half_lld":
            if self.lod is None:
                raise ValueError("half_lld needs the lod array")
            D = np.asarray(self.lod, float)
            if D.shape != X.shape:
                raise ValueError(
                    f"lod shape {D.shape} does not match X {X.shape}; the caller "
                    "must slice lod to the same rows as X.")
            with np.errstate(divide="ignore", invalid="ignore"):
                half = np.log10(np.where(D > 0, D / 2.0, np.nan))
            out = np.where(np.isfinite(L), L, half)
            return np.where(np.isfinite(out), out, self.train_median_)
        raise ValueError(f"unknown imputation strategy {self.strategy!r}")


def regressors():
    """Name -> (pipeline, param grid). Ridge is the interpretable floor; the boosted
    tree is the capable model. Both are reported, because a gap that only one of them
    shows is a property of that estimator and not of the protocol."""
    return {
        "hist_gbr": (
            Pipeline([("scale", StandardScaler(with_mean=True)),
                      ("est", HistGradientBoostingRegressor(random_state=SEED))]),
            {"est__max_depth": [None, 6],
             "est__learning_rate": [0.05, 0.1],
             "est__max_iter": [300]},
        ),
        "ridge": (
            Pipeline([("scale", StandardScaler()),
                      ("est", Ridge(random_state=None))]),
            {"est__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
    }


def classifiers():
    return {
        "random_forest": (
            Pipeline([("est", RandomForestClassifier(
                n_estimators=500, random_state=SEED, n_jobs=N_JOBS,
                class_weight="balanced"))]),
            {"est__max_depth": [None, 8],
             "est__min_samples_leaf": [1, 5]},
        ),
        "hist_gbc": (
            Pipeline([("est", HistGradientBoostingClassifier(random_state=SEED))]),
            {"est__max_depth": [None, 6], "est__learning_rate": [0.05, 0.1],
             "est__max_iter": [300]},
        ),
        "logistic": (
            Pipeline([("scale", StandardScaler(with_mean=False)),
                      ("est", LogisticRegression(
                          max_iter=2000, class_weight="balanced",
                          random_state=SEED))]),
            {"est__C": [0.01, 0.1, 1.0, 10.0]},
        ),
    }


def fit_nested(pipe, grid, X, y, kind="reg", inner_k=INNER_K, scoring=None):
    """Fit with hyperparameters chosen on an inner split of the training data only.

    Returns (fitted_estimator, best_params, inner_score). Degrades to a plain fit when
    the training fold is too small to split, and says so in best_params rather than
    silently skipping the inner loop.
    """
    n = len(y)
    if kind == "clf":
        counts = np.unique(y, return_counts=True)[1]
        k = int(min(inner_k, counts.min())) if counts.size else 0
        if k < 2:
            pipe.fit(X, y)
            return pipe, {"_inner_cv": "skipped_min_class_too_small"}, np.nan
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        scoring = scoring or "balanced_accuracy"
    else:
        k = int(min(inner_k, max(2, n // 20)))
        if n < 20:
            pipe.fit(X, y)
            return pipe, {"_inner_cv": "skipped_training_fold_too_small"}, np.nan
        cv = KFold(n_splits=k, shuffle=True, random_state=SEED)
        scoring = scoring or "neg_root_mean_squared_error"
    gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=N_JOBS,
                      refit=True, error_score="raise")
    gs.fit(X, y)
    return gs.best_estimator_, dict(gs.best_params_), float(gs.best_score_)


def categorical_design(cov_frame, train_idx):
    """One-hot Koppen group and GLiM class, categories learned on the training fold.

    A category seen only in the test fold becomes all-zeros rather than a new column,
    which is the honest behaviour: the model genuinely has not seen it. Under LOSO
    this happens often, and it is part of what LOSO is measuring.
    """
    cols = ["kg_group", "glim_class"]
    Xdf = cov_frame[cols].astype(str).fillna("missing")
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(Xdf.iloc[train_idx])
    return enc.transform(Xdf), enc.get_feature_names_out(cols).tolist()
