"""Paths, seed and the protocol registry. Single source of truth: nothing else in
the harness hardcodes a path, a fold column or a value of K.
"""
import os

SEED = 20260925

# Paths default to the repository layout and can be overridden by environment
# variables, so the harness runs unchanged on any machine.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INTERIM = os.environ.get("GEOSHIFT_INTERIM", os.path.join(_REPO, "data", "interim"))
PREREG = os.environ.get("GEOSHIFT_FOLDS", os.path.join(_REPO, "folds"))
CROSSWALK = os.environ.get("GEOSHIFT_METADATA", os.path.join(_REPO, "metadata"))
TELLUS_DIR = os.environ.get("GEOSHIFT_TELLUS", os.path.join(_REPO, "data", "tellus"))
TELLUS_XLSX = os.path.join(
    TELLUS_DIR, "58xxxxA-65xxxxA_Shallow_Topsoil_Download_v1.2.xlsx")
RESULTS = os.environ.get("GEOSHIFT_RESULTS", os.path.join(_REPO, "results"))

SURVEYS = ["GEMAS", "NGSA", "NASGL"]
FIVE_SET_ASIDE = ["Li", "In", "Te", "Tl", "Se"]     # secondary Task 5 only

# Protocol registry. `file` is the fold CSV, `column` the fold-id column inside it.
# `task2` records whether the fold definition passed the eligibility gate in
# build_fold_definitions_v2.py -- all three surveys present at n >= 30 in EVERY fold.
# The harness refuses to run Task 2 on a protocol marked False, rather than trusting
# the caller to remember which ones were excluded.
PROTOCOLS = {
    "random_k10": dict(file="folds-random-kfold.csv", column="fold_random_kfold",
                       k=10, task2=True, headline=False),
    "random_k3": dict(file="folds-random-kfold.csv", column="fold_random_k3",
                      k=3, task2=True, headline=True),
    "spatial_k3": dict(file="folds-spatial-block.csv", column="fold_spatial_block",
                       k=3, task2=True, headline=True),
    "environmental_k10": dict(file="folds-environmental.csv",
                              column="fold_environmental",
                              k=10, task2=False, headline=False),
    "environmental_k3": dict(file="folds-environmental.csv",
                             column="fold_environmental_k3",
                             k=3, task2=True, headline=True),
    "loso": dict(file="folds-loso.csv", column="fold_loso",
                 k=3, task2=False, headline=True),
    "predictor_kmeans": dict(file="folds-predictor-kmeans-sensitivity.csv",
                             column="fold_predictor_kmeans",
                             k=10, task2=False, headline=False),
}

# The headline protocol comparison is run at matched K = 3 (methodology 3.3): at
# K = 10 a model trains on 90% of the data and at K = 3 on 67%, so comparing across
# different K confounds the blocking axis with training-set size.
HEADLINE = [p for p, c in PROTOCOLS.items() if c["headline"]]
MATCHED_K = 3

TASK2P_FILE = "folds-task2p.csv"
TASK2P_PAIRS = [("GEMAS", "NGSA"), ("GEMAS", "NASGL"), ("NGSA", "NASGL")]

# Task 3 exists on one survey pair only: GEMAS publishes a detection limit for 15 of
# 53 elements, so the disparity ratio is computable on 34 core elements for
# NGSA<->NASGL, 11 for GEMAS<->NGSA and 3 for GEMAS<->NASGL.
TASK3_PAIR = ("NGSA", "NASGL")

MIN_PER_SURVEY_PER_FOLD = 30
