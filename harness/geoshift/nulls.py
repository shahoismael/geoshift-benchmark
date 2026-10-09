"""Model-free nulls for the classification tasks, and why they are adversarial.

A learned classifier that reads a censoring mask and reports 0.95 balanced accuracy
has demonstrated nothing until it beats a rule that reads ONE column of that mask.
On Tellus a single element's indicator reaches 0.996 (Hg), 0.974 (Ag), 0.876 (W).
So the acceptance criterion, fixed before any model was fitted, is that the model
must beat the best single-element rule -- not chance, and not the majority class.

All three nulls are fitted INSIDE the training fold and scored on the test fold,
exactly like the model. A null chosen on the full dataset would be optimistically
biased in the null's favour, which would understate the model and is the mirror image
of the usual leak.
"""
from __future__ import annotations
import numpy as np

from .metrics import balanced_accuracy


def majority_class(y_train, y_test):
    """Predict the training fold's most frequent class for everything."""
    vals, counts = np.unique(y_train, return_counts=True)
    pred = np.full(len(y_test), vals[np.argmax(counts)])
    return balanced_accuracy(y_test, pred), dict(rule=str(vals[np.argmax(counts)]))


def mask_density(M_train, y_train, M_test, y_test, n_thresholds=200):
    """One feature: the fraction of a sample's elements that are censored.

    Thresholds are swept on the TRAINING fold only. For more than two classes the
    density axis is cut into as many intervals as there are classes, with cut points
    chosen to maximise training balanced accuracy over a grid.
    """
    d_tr = np.asarray(M_train, float).mean(axis=1)
    d_te = np.asarray(M_test, float).mean(axis=1)
    classes = np.unique(y_train)
    if classes.size == 2:
        lo, hi = np.percentile(d_tr, [1, 99])
        grid = np.linspace(lo, hi, n_thresholds)
        best = (-1.0, None, None)
        for t in grid:
            for order in (0, 1):
                a, b = (classes[0], classes[1]) if order == 0 else (classes[1], classes[0])
                pred = np.where(d_tr <= t, a, b)
                s = balanced_accuracy(y_train, pred)
                if s is not None and np.isfinite(s) and s > best[0]:
                    best = (s, t, (a, b))
        _, t, (a, b) = best
        pred_te = np.where(d_te <= t, a, b)
        return balanced_accuracy(y_test, pred_te), dict(threshold=float(t),
                                                        low_class=str(a),
                                                        high_class=str(b))
    # >2 classes: order classes by their training median density and cut at the
    # midpoints between consecutive medians. Simple, deterministic, and still a
    # one-feature rule.
    med = {c: float(np.median(d_tr[y_train == c])) for c in classes}
    order = sorted(classes, key=lambda c: med[c])
    cuts = [(med[order[i]] + med[order[i + 1]]) / 2.0 for i in range(len(order) - 1)]
    pred_te = np.array([order[int(np.searchsorted(cuts, v))] for v in d_te])
    return balanced_accuracy(y_test, pred_te), dict(
        cuts=[float(c) for c in cuts], order=[str(c) for c in order])


def best_single_element(M_train, y_train, M_test, y_test, element_names):
    """The binding null: the single element whose censoring indicator best separates
    the classes on the training fold, then applied unchanged to the test fold.

    For each element the indicator takes two values, so the rule is a mapping from
    {0, 1} to classes, chosen on the training fold as the majority class within each
    indicator value. That is the strongest one-column rule available.
    """
    M_train = np.asarray(M_train, np.int8)
    M_test = np.asarray(M_test, np.int8)
    y_train = np.asarray(y_train)
    best = (-1.0, None, None)
    for j in range(M_train.shape[1]):
        col = M_train[:, j]
        mapping = {}
        for v in (0, 1):
            sel = col == v
            if sel.sum() == 0:
                mapping[v] = None
                continue
            vals, counts = np.unique(y_train[sel], return_counts=True)
            mapping[v] = vals[np.argmax(counts)]
        if mapping[0] is None or mapping[1] is None:
            continue                      # constant column carries no rule
        pred_tr = np.where(col == 0, mapping[0], mapping[1])
        s = balanced_accuracy(y_train, pred_tr)
        if s is not None and np.isfinite(s) and s > best[0]:
            best = (s, j, mapping)
    score_tr, j, mapping = best
    if j is None:
        return np.nan, dict(element=None,
                            note="every element indicator constant in the "
                                 "training fold; no single-element rule exists")
    col_te = M_test[:, j]
    pred_te = np.where(col_te == 0, mapping[0], mapping[1])
    return balanced_accuracy(y_test, pred_te), dict(
        element=str(element_names[j]),
        train_balanced_accuracy=float(score_tr),
        maps_0_to=str(mapping[0]), maps_1_to=str(mapping[1]))


def all_nulls(M_train, y_train, M_test, y_test, element_names):
    """Returns a flat dict of the three nulls, plus the binding one and its value.

    `binding_null` is the maximum over the three. The model's acceptance criterion is
    stated against THIS number, not against chance.
    """
    s_maj, d_maj = majority_class(y_train, y_test)
    s_den, d_den = mask_density(M_train, y_train, M_test, y_test)
    s_bse, d_bse = best_single_element(M_train, y_train, M_test, y_test,
                                       element_names)
    scores = {"majority_class": s_maj, "mask_density": s_den,
              "best_single_element": s_bse}
    finite = {k: v for k, v in scores.items() if v is not None and np.isfinite(v)}
    binding = max(finite, key=finite.get) if finite else None
    return dict(
        null_majority_class=s_maj,
        null_mask_density=s_den,
        null_best_single_element=s_bse,
        null_best_single_element_which=d_bse.get("element"),
        null_mask_density_detail=str(d_den),
        binding_null=binding,
        binding_null_value=(finite[binding] if binding else np.nan),
    )
