#!/usr/bin/env python
"""GeoShift evaluation harness -- command-line entry point.

Every run writes a provenance sidecar next to its results: seed, library versions,
input file hashes, timestamp and the exact arguments. Any number that reaches the
manuscript can therefore be traced to the run that produced it, and a number whose
sidecar does not match the locked input hashes is not a result.

NOTHING IS RUN BY DEFAULT. With no task, control or --all, the command performs the
preflight check and stops.

Examples (PowerShell):
  python run_geoshift.py --check
  python run_geoshift.py --occupancy
  python run_geoshift.py --task 4
  python run_geoshift.py --task 2 --protocol random_k3
  python run_geoshift.py --task 1 --protocol loso --elements 3
  python run_geoshift.py --control loa
  python run_geoshift.py --control permutation --n-perm 50
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, platform, sys, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoshift import METHODOLOGY_SHA256, __version__
from geoshift.config import (PROTOCOLS, HEADLINE, PREREG, CROSSWALK, INTERIM,
                             TELLUS_XLSX, RESULTS, SEED, MATCHED_K)
from geoshift import tasks, controls
from geoshift.data import load_observational, core_elements
from geoshift.folds import fold_occupancy


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def provenance(args):
    inputs = {}
    for d in (PREREG, CROSSWALK):
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            p = os.path.join(d, name)
            if os.path.isfile(p) and name.lower().endswith((".csv", ".md")):
                inputs[name] = dict(bytes=os.path.getsize(p), sha256=sha256(p))
    import sklearn, scipy
    return dict(
        harness_version=__version__,
        methodology_sha256=METHODOLOGY_SHA256,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seed=SEED, argv=sys.argv[1:], args=vars(args),
        python=sys.version.split()[0], platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__,
        scikit_learn=sklearn.__version__, scipy=scipy.__version__,
        inputs=inputs)


def write(rows, name, args):
    os.makedirs(RESULTS, exist_ok=True)
    if rows is None or (hasattr(rows, "__len__") and len(rows) == 0):
        print(f"  {name}: no rows produced -- nothing written")
        return None
    df = pd.DataFrame(rows)
    f = os.path.join(RESULTS, f"{name}.csv")
    df.to_csv(f, index=False, encoding="utf-8")
    with open(os.path.join(RESULTS, f"{name}.provenance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(provenance(args), fh, indent=2)
    print(f"  wrote {f}  ({len(df)} rows)")
    return df


def preflight():
    print("=" * 76)
    print(f"PREFLIGHT -- harness {__version__}, methodology {METHODOLOGY_SHA256[:16]}...")
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK  ' if cond else 'MISS'}] {label:34s} {detail}")

    for s in ["gemas", "ngsa", "nasgl"]:
        for kind in ["meta", "conc", "mask", "lod"]:
            p = os.path.join(INTERIM, f"{s}_{kind}.parquet")
            check(f"{s}_{kind}.parquet", os.path.exists(p), p if not os.path.exists(p) else "")
    for name in ["folds-random-kfold.csv", "folds-loso.csv", "folds-spatial-block.csv",
                 "folds-environmental.csv", "folds-task2p.csv",
                 "primary-core-elements.csv", "autocorrelation-block-size.csv"]:
        p = os.path.join(PREREG, name)
        check(name, os.path.exists(p))
    for name in ["sample-environmental-strata.csv", "lod-disparity-delta.csv"]:
        p = os.path.join(CROSSWALK, name)
        check(name, os.path.exists(p))
    check("tellus workbook", os.path.exists(TELLUS_XLSX))

    if ok:
        try:
            els = core_elements()
            meta, conc, mask, lod = load_observational(els)
            print(f"  core elements : {len(els)}")
            print(f"  samples       : {len(meta):,}")
            print(f"  censored cells: {int(mask.to_numpy().sum()):,} "
                  f"({100 * mask.to_numpy().mean():.2f}%)")
            bs = pd.read_csv(os.path.join(PREREG, "autocorrelation-block-size.csv"))
            print(f"  spatial block : {float(bs.block_side_km.iloc[0]):.0f} km")
        except Exception as e:                                   # noqa: BLE001
            print(f"  [FAIL] loading data: {type(e).__name__}: {e}")
            ok = False
    print(f"  results dir   : {RESULTS}")
    print("  VERDICT:", "ready" if ok else "NOT ready -- fix the MISS rows above")
    return ok


def occupancy(args):
    els = core_elements()
    meta, *_ = load_observational(els)
    rows = []
    print("\nfold occupancy and Task 2 eligibility, recomputed from the data:")
    for p in PROTOCOLS:
        try:
            v, t = fold_occupancy(p, meta)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {p:20s} ERROR {type(e).__name__}: {e}")
            continue
        rows.append(v)
        print(f"  {p:20s} {v['n_folds']:3d} folds  all3>=n30 in "
              f"{v['folds_all_surveys_at_min']:3d}  "
              f"{'ELIGIBLE' if v['task2_eligible'] else 'NOT ELIGIBLE'}"
              + ("" if v["task2_eligible"] == v["registry_says_task2"]
                 else "   <-- DISAGREES WITH REGISTRY"))
    print(f"\n  headline comparison runs at matched K = {MATCHED_K}: "
          f"{', '.join(HEADLINE)}")
    return write(rows, "fold_occupancy", args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=[],
                    choices=["1", "2", "2p", "3", "4", "4b"])
    ap.add_argument("--protocol", action="append", default=[],
                    choices=sorted(PROTOCOLS))
    ap.add_argument("--control", action="append", default=[],
                    choices=["permutation", "loa"])
    ap.add_argument("--model", default=None,
                    help="regressor for task 1/3 or classifier for 2/2p/4")
    ap.add_argument("--imputation", default="half_lld",
                    choices=["half_lld", "nan", "median_train", "maskreg"])
    ap.add_argument("--covariates", default="environmental",
                    choices=["environmental", "coords", "none"])
    ap.add_argument("--elements", type=int, default=0,
                    help="limit to the first N core elements (SMOKE TEST ONLY)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--perm-part", type=int, default=0,
                    help="1..N: run this part of a split permutation control. The "
                         "train/test split and the observed accuracy are identical in "
                         "every part; only the permutation draws differ, so the parts "
                         "pool into one null distribution.")
    ap.add_argument("--headline", action="store_true",
                    help="use the matched-K headline protocol set")
    ap.add_argument("--occupancy", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check or not (args.task or args.control or args.occupancy):
        preflight()
        if args.check or not (args.task or args.control or args.occupancy):
            return
    if not preflight():
        sys.exit(1)
    if args.occupancy:
        occupancy(args)
        if not (args.task or args.control):
            return

    els = core_elements()
    if args.elements:
        els = els[:args.elements]
        print(f"\n!! SMOKE TEST: {len(els)} of {len(core_elements())} elements. "
              "NOT A RESULT. Do not quote any number from this run.\n")

    protos = args.protocol or (HEADLINE if args.headline else [])
    for t in args.task:
        print(f"\n### TASK {t}")
        if t == "1":
            if not protos:
                print("  --task 1 needs --protocol or --headline"); continue
            for p in protos:
                print(f"  protocol {p}")
                write(tasks.task1(p, els, model=args.model or "hist_gbr",
                                  imputation=args.imputation,
                                  covariates=args.covariates),
                      f"task1_{p}_{args.imputation}_{args.covariates}", args)
        elif t == "2":
            if not protos:
                print("  --task 2 needs --protocol or --headline"); continue
            for p in protos:
                print(f"  protocol {p}")
                try:
                    write(tasks.task2(p, els, model=args.model or "random_forest"),
                          f"task2_{p}", args)
                except ValueError as e:
                    print(f"  REFUSED: {e}")
        elif t == "2p":
            write(tasks.task2p(els, model=args.model or "random_forest"),
                  "task2p_pairwise", args)
        elif t == "3":
            rows, slopes = tasks.task3(els, model=args.model or "hist_gbr",
                                       imputation=args.imputation)
            write(rows, "task3_per_element", args)
            write(slopes, "task3_slopes", args)
        elif t in ("4", "4b"):
            write(tasks.task4(extended=(t == "4b"),
                              model=args.model or "random_forest"),
                  f"task{t}_route_identification", args)

    for c in args.control:
        print(f"\n### CONTROL {c}")
        if c == "permutation":
            print(f"  fitting the classifier {args.n_perm + 1} times; this is the "
                  "slowest thing the harness does")
            part = args.perm_part
            tag = f"_part{part}" if part else ""
            pseed = (SEED + 100000 * part) if part else None
            power, null = controls.stratified_permutation_test(
                n_perm=args.n_perm, elements=els,
                model=args.model or "random_forest", perm_seed=pseed)
            if part:
                power["perm_part"] = part
                power["perm_seed"] = pseed
                power["p_value_note"] = (
                    "PARTIAL. This p-value uses only this part's permutations. Pool "
                    "every part with pool_permutation_parts.py before quoting a "
                    "p-value.")
            write([power], f"control_permutation{tag}", args)
            for k in ("multi_survey_strata", "pct_samples_in_multi",
                      "log_permutations", "observed_balanced_accuracy", "verdict"):
                print(f"    {k}: {power.get(k)}")
            if null is not None:
                write(pd.DataFrame({"permutation_balanced_accuracy": null}),
                      f"control_permutation_null_distribution{tag}", args)
        elif c == "loa":
            write(controls.paired_limits_of_agreement(),
                  "control_limits_of_agreement", args)

    print(f"\ndone. results in {RESULTS}")


if __name__ == "__main__":
    main()
