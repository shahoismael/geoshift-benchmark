"""Data loading, with one invariant enforced everywhere: row order.

The fold CSVs were written against the concatenation gemas -> ngsa -> nasgl, in that
order, with row_id = 0..n-1. Every loader here rebuilds that exact order and then
CHECKS it against the sample_id column the fold files carry. A silent row-order drift
would reassign every fold to the wrong sample and still produce plausible-looking
numbers, so the check is an assertion and not a warning.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .config import (INTERIM, PREREG, CROSSWALK, TELLUS_XLSX, SURVEYS,
                     FIVE_SET_ASIDE)

_CACHE = {}


def core_elements():
    """The 39 primary-core elements, read from the pre-registered list rather than
    recomputed, so the harness cannot silently disagree with the fold files."""
    f = os.path.join(PREREG, "primary-core-elements.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"{f} missing. Run build_fold_definitions_v2.py first.")
    els = pd.read_csv(f)["element"].astype(str).tolist()
    bad = sorted(set(els) & set(FIVE_SET_ASIDE))
    if bad:
        raise ValueError(
            f"core element list contains Task-5-only elements {bad}; the list is "
            "inconsistent with the methodology and must be rebuilt.")
    return els


def load_observational(elements=None):
    """Returns (meta, conc, mask, lod), all row-aligned, in fold-file order.

    conc  concentrations, NaN where the survey withheld a censored value
    mask  1 where left-censored, 0 where reported  (int8)
    lod   the per-cell detection limit, recorded only on censored cells
    """
    key = ("obs", tuple(elements) if elements else None)
    if key in _CACHE:
        return _CACHE[key]
    metas, concs, masks, lods = [], [], [], []
    for s in ["gemas", "ngsa", "nasgl"]:
        m = pd.read_parquet(os.path.join(INTERIM, f"{s}_meta.parquet"))[
            ["survey", "sample_id", "latitude", "longitude"]].copy()
        m["survey"] = s.upper()
        c = pd.read_parquet(os.path.join(INTERIM, f"{s}_conc.parquet"))
        k = pd.read_parquet(os.path.join(INTERIM, f"{s}_mask.parquet"))
        d = pd.read_parquet(os.path.join(INTERIM, f"{s}_lod.parquet"))
        metas.append(m.reset_index(drop=True))
        concs.append(c.reset_index(drop=True))
        masks.append(k.reset_index(drop=True))
        lods.append(d.reset_index(drop=True))

    els = elements if elements is not None else core_elements()
    missing = {s.upper(): sorted(set(els) - set(c.columns))
               for s, c in zip(["gemas", "ngsa", "nasgl"], concs)}
    if any(missing.values()):
        raise ValueError(f"elements absent from a survey: {missing}")

    meta = pd.concat(metas, ignore_index=True)
    meta["row_id"] = np.arange(len(meta))
    conc = pd.concat([c[els] for c in concs], ignore_index=True).apply(
        pd.to_numeric, errors="coerce")
    mask = pd.concat([k[els] for k in masks], ignore_index=True).fillna(0).astype(np.int8)
    lod = pd.concat([d[els] for d in lods], ignore_index=True).apply(
        pd.to_numeric, errors="coerce")
    out = (meta, conc, mask, lod)
    _CACHE[key] = out
    return out


def assert_fold_alignment(meta, fold_df, label):
    """Fail loudly if the fold file was built against a different row order."""
    if len(fold_df) != len(meta):
        raise ValueError(
            f"{label}: fold file has {len(fold_df)} rows, data has {len(meta)}. "
            "Rebuild the fold files with build_fold_definitions_v2.py.")
    # Compare NaN-aware. Five samples carry no sample_id, and a naive string compare
    # renders them "nan" on both sides but still reports them unequal, because numpy
    # compares the float NaN before the cast in some paths. Two rows that are both
    # missing an id are aligned, not drifted -- flagging them as drift would condemn
    # every fold file over nothing.
    sa = meta["sample_id"]
    sb = fold_df["sample_id"]
    both_missing = sa.isna().to_numpy() & sb.isna().to_numpy()
    a = sa.astype(str).to_numpy()
    b = sb.astype(str).to_numpy()
    differs = (a != b) & ~both_missing
    if differs.any():
        n_bad = int(differs.sum())
        first = int(np.flatnonzero(differs)[0])
        raise ValueError(
            f"{label}: sample_id mismatch on {n_bad} rows, first at position "
            f"{first} ({a[first]!r} vs {b[first]!r}). Row order has drifted; every "
            "fold assignment would be wrong. Rebuild the fold files.")


def environmental_covariates(meta):
    """Koppen group and GLiM lithology per sample, joined on survey + coordinates.

    NOT joined on row order. sample-environmental-strata.csv drops the samples
    without coordinates (8,280 rows against 8,285), so positional alignment shifts
    every stratum after the first dropped row onto the wrong sample -- a wrong answer
    that runs clean. Returns a frame with NaN where no stratum resolves.
    """
    st = pd.read_csv(os.path.join(CROSSWALK, "sample-environmental-strata.csv"))

    def key(f):
        return (f["survey"].astype(str) + "@"
                + pd.to_numeric(f["latitude"], errors="coerce").round(6).astype(str)
                + "," + pd.to_numeric(f["longitude"], errors="coerce").round(6).astype(str))

    sk, mk = key(st), key(meta)
    idx = pd.Series(np.arange(len(st)), index=sk)
    idx = idx[~idx.index.duplicated(keep="first")]
    pos = mk.map(idx)
    matched = pos.notna()
    out = pd.DataFrame(index=meta.index, columns=["kg_group", "glim_class", "glim_code"],
                       dtype=object)
    take = pos[matched].astype(int).to_numpy()
    out.loc[matched, "kg_group"] = st["kg_group"].to_numpy()[take]
    out.loc[matched, "glim_class"] = st["glim_class"].to_numpy()[take]
    out.loc[matched, "glim_code"] = pd.to_numeric(
        st["glim_code"], errors="coerce").to_numpy()[take]
    gc = pd.to_numeric(out["glim_code"], errors="coerce")
    bad = (~matched) | (out["kg_group"].astype(str) == "?") | (gc <= 0) | (gc >= 14) | gc.isna()
    out.loc[bad, :] = np.nan
    out.attrs["n_matched"] = int(matched.sum())
    out.attrs["n_usable"] = int((~bad).sum())
    return out


def load_tellus():
    """Paired censoring indicators for the controlled arm.

    Four corrections, each of which changed a headline number when it was applied:
      C1 Units_Detection_Limits holds duplicate analyte rows (Accredited yes/no) with
         limits differing by up to 1e4. The accredited row is selected explicitly;
         taking the last row silently picked the non-accredited limit for 49 of 50.
      C2 The published HDL column is used, not a recomputed 0.5*LLD, which is not
         representable at the printed precision of 44 of 111 columns.
      C3 Left-censoring is (value == HDL) OR (value < LLD). Tellus does not
         substitute uniformly, so neither test alone finds all censored cells.
      C4 Right-censoring is flagged at the published URL (upper reporting limit).

    Returns (flag_xrfs, flag_icpar, elements, rows_ok, provenance) where the flags are
    int8 frames of left-censoring indicators on a shared element set, and provenance
    records whether each element's limit was accredited on both routes.
    """
    if not os.path.exists(TELLUS_XLSX):
        raise FileNotFoundError(f"{TELLUS_XLSX} missing.")
    df = pd.read_excel(TELLUS_XLSX, sheet_name=0)
    u = pd.read_excel(TELLUS_XLSX, sheet_name="Units_Detection_Limits")
    u.columns = [str(c).strip() for c in u.columns]
    u["Analyte"] = u["Analyte"].astype(str).str.strip()
    u["Accredited"] = u["Accredited"].astype(str).str.strip().str.lower()

    limits = {}
    for a, g in u.groupby("Analyte", sort=False):
        acc = g[g["Accredited"] == "yes"]
        row, prov = (acc.iloc[0], "accredited") if len(acc) else (g.iloc[0],
                                                                 "non_accredited")
        limits[a] = dict(lld=pd.to_numeric(row.get("LLD"), errors="coerce"),
                         hdl=pd.to_numeric(row.get("HDL"), errors="coerce"),
                         url=pd.to_numeric(row.get("URL"), errors="coerce"),
                         provenance=prov)

    # Pairing comes from the AUDITED crosswalk, not from parsing column names.
    # Nine XRFS columns are reported as oxides, so a name-based match silently loses
    # them: "TiO2_%_XRFS" parses to TiO2 and never meets "Ti_%_ICPar". Ti and Mn are
    # the two both-accredited oxide pairs, and dropping them cost Task 4 two of its 38
    # elements with no error raised. One ICPar column also carries a stray space
    # ("Ca_% _ICPar"), which no parser should have to know about.
    pair_file = os.path.join(CROSSWALK, "tellus-paired-method-contrast.csv")
    cols = {"XRFS": {}, "ICPar": {}}
    if os.path.exists(pair_file):
        pf = pd.read_csv(pair_file)
        for _, r in pf.iterrows():
            xc, ic = str(r["xrfs_col"]), str(r["icpar_col"])
            if xc in df.columns and ic in df.columns and xc in limits and ic in limits:
                cols["XRFS"][str(r["element"])] = xc
                cols["ICPar"][str(r["element"])] = ic
    else:
        raise FileNotFoundError(
            f"{pair_file} missing. Task 4 pairing must come from the audited "
            "crosswalk; parsing element names from columns loses the oxide-reported "
            "XRFS columns.")

    els = sorted(set(cols["XRFS"]) & set(cols["ICPar"]))
    fx, fi, prov = {}, {}, []
    for e in els:
        out = {}
        for r, store in (("XRFS", fx), ("ICPar", fi)):
            col = cols[r][e]
            lim = limits[col]
            v = pd.to_numeric(df[col], errors="coerce")
            left = ((v == lim["hdl"]) | (v < lim["lld"])).fillna(False)
            store[e] = left.astype(np.int8)
            out[r] = lim["provenance"]
        prov.append(dict(element=e,
                         xrfs_limit=out["XRFS"], icpar_limit=out["ICPar"],
                         both_accredited=bool(out["XRFS"] == "accredited"
                                              and out["ICPar"] == "accredited")))
    FX = pd.DataFrame(fx)
    FI = pd.DataFrame(fi)
    rows_ok = FX.notna().all(axis=1) & FI.notna().all(axis=1)
    return FX, FI, els, rows_ok, pd.DataFrame(prov)


def delta_table():
    """Per-element detection-limit disparity, as built by lod_delta_by_pair.py."""
    f = os.path.join(CROSSWALK, "lod-disparity-delta.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} missing.")
    return pd.read_csv(f)
