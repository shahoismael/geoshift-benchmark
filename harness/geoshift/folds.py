"""Fold access. The harness READS folds; it never invents one.

Every protocol resolves to a pre-registered CSV and a column inside it, listed in
config.PROTOCOLS. Two gates are enforced here rather than left to the caller:

  1. Row alignment against the data (data.assert_fold_alignment).
  2. Task 2 eligibility -- all three surveys present at n >= 30 in EVERY test fold.
     LOSO fails this by construction; environmental_k10 and predictor_kmeans fail it
     on measured counts. Asking for Task 2 on one of those raises rather than
     returning a number that looks like a result.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .config import (PREREG, PROTOCOLS, TASK2P_FILE, TASK2P_PAIRS,
                     MIN_PER_SURVEY_PER_FOLD)
from .data import assert_fold_alignment


def _read(protocol):
    if protocol not in PROTOCOLS:
        raise KeyError(f"unknown protocol {protocol!r}; "
                       f"known: {sorted(PROTOCOLS)}")
    spec = PROTOCOLS[protocol]
    path = os.path.join(PREREG, spec["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run build_fold_definitions_v2.py.")
    df = pd.read_csv(path)
    if spec["column"] not in df.columns:
        raise ValueError(
            f"column {spec['column']!r} absent from {spec['file']}. The fold files "
            "predate the matched-K amendment; rebuild them.")
    return df, spec


def folds(protocol, meta):
    """Yields (fold_id, train_idx, test_idx) as positional index arrays."""
    df, spec = _read(protocol)
    assert_fold_alignment(meta, df, f"protocol {protocol}")
    f = pd.to_numeric(df[spec["column"]], errors="coerce")
    usable = f.notna().to_numpy()
    n_dropped = int((~usable).sum())
    ids = sorted(int(v) for v in f[usable].unique())
    if len(ids) != spec["k"]:
        # Not fatal, but it means the file disagrees with the registry -- report it.
        print(f"  note: {protocol} has {len(ids)} folds, registry says "
              f"{spec['k']}")
    arr = f.to_numpy()
    for fid in ids:
        test = np.flatnonzero(usable & (arr == fid))
        train = np.flatnonzero(usable & (arr != fid))
        yield fid, train, test
    if n_dropped:
        print(f"  note: {protocol} leaves {n_dropped} samples unassigned "
              "(no coordinates or no resolvable stratum); they appear in no fold.")


def fold_occupancy(protocol, meta):
    """Per-fold survey counts, plus the Task 2 verdict recomputed from the data."""
    df, spec = _read(protocol)
    assert_fold_alignment(meta, df, f"protocol {protocol}")
    f = pd.to_numeric(df[spec["column"]], errors="coerce")
    t = pd.crosstab(f, meta["survey"])
    n_folds = len(t)
    all_present = int(t.gt(0).all(axis=1).sum())
    all_enough = int(t.ge(MIN_PER_SURVEY_PER_FOLD).all(axis=1).sum())
    return dict(protocol=protocol, n_folds=n_folds,
                folds_all_surveys_present=all_present,
                folds_all_surveys_at_min=all_enough,
                min_per_survey_threshold=MIN_PER_SURVEY_PER_FOLD,
                task2_eligible=bool(all_enough == n_folds and n_folds >= 2),
                registry_says_task2=bool(spec["task2"])), t


def require_task2_eligible(protocol, meta):
    """Raise unless this protocol can carry Task 2. Checked against the data, not
    against the registry alone, so an edited fold file cannot smuggle a protocol
    back in."""
    verdict, table = fold_occupancy(protocol, meta)
    if not verdict["task2_eligible"]:
        short = table[~table.ge(MIN_PER_SURVEY_PER_FOLD).all(axis=1)]
        detail = "; ".join(
            f"fold {i}: " + ", ".join(
                f"{c}={int(r[c])}" for c in table.columns
                if r[c] < MIN_PER_SURVEY_PER_FOLD)
            for i, r in short.iterrows())
        raise ValueError(
            f"Task 2 is not defined under {protocol}: not every fold holds "
            f"{MIN_PER_SURVEY_PER_FOLD}+ samples of all three surveys. {detail}. "
            "This is the same defect that removed Task 2 from LOSO (methodology "
            "3.1); no balanced accuracy is reported here.")
    if verdict["task2_eligible"] != verdict["registry_says_task2"]:
        print(f"  warning: {protocol} eligibility recomputed as "
              f"{verdict['task2_eligible']} but the registry says "
              f"{verdict['registry_says_task2']}. Trusting the data. Update "
              "config.PROTOCOLS.")
    return verdict


def task2p_splits(meta):
    """Yields (pair_label, train_idx, test_idx) for the pairwise site holdouts.

    Task 2p exists because LOSO cannot carry Task 2: it holds out SITES within a
    survey pair, so both classes are present in the test set and chance is 1/2.
    """
    path = os.path.join(PREREG, TASK2P_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run build_fold_definitions_v2.py.")
    df = pd.read_csv(path)
    assert_fold_alignment(meta, df, "task2p")
    for a, b in TASK2P_PAIRS:
        col = f"split_{a}_{b}"
        if col not in df.columns:
            raise ValueError(f"column {col!r} absent from {TASK2P_FILE}.")
        # Samples outside the pair carry no split label. With pandas' nullable
        # "string" dtype a comparison against NA yields NA rather than False, and
        # flatnonzero cannot evaluate that. Fill the blanks first so the comparison
        # returns a plain boolean array.
        s = df[col].astype("string").fillna("")
        train = np.flatnonzero((s == "train").to_numpy(dtype=bool))
        test = np.flatnonzero((s == "test").to_numpy(dtype=bool))
        n_cls = meta["survey"].to_numpy()[test]
        if len(np.unique(n_cls)) != 2:
            raise ValueError(
                f"{col}: test set holds {len(np.unique(n_cls))} classes, expected 2. "
                "The split file is broken; rebuild it.")
        yield f"{a}_{b}", train, test


def task3_pair_mask(meta, pair):
    """Positional indices of the two surveys in the Task 3 pair, as two arrays.

    Task 3 is a single survey pair because the disparity ratio needs a published
    limit in both surveys, and GEMAS publishes one for 15 of 53 elements.
    """
    a, b = pair
    s = meta["survey"].to_numpy()
    return np.flatnonzero(s == a), np.flatnonzero(s == b)
