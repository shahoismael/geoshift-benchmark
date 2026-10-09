"""Rebuild every fold definition on the v7 amended design. Supersedes
build_fold_definitions.py, which stays on disk as the superseded record.

Fold assignment is a DESIGN choice, not a model fit: it uses coordinates, survey
labels, environmental strata and (for one sensitivity column only) predictor values,
but never the target of any task, so computing it on the full table introduces no
train/test leakage. All randomness is seeded.

WHAT CHANGED FROM v1, AND WHY
  spatial     the 2 degree grid is withdrawn as unjustified. Blocks are now squares
              of side b km, b read from autocorrelation-block-size.csv, which
              measure_autocorrelation_range.py must produce first. Blocks are
              assigned to folds by a deterministic greedy balance, not modulo, so
              fold sizes do not depend on block ordering.
  environmental  v1 ran k-means on log concentrations. That blocks on the
              MEASUREMENTS, not on the environment, and risks blocking partly on the
              censoring structure Tasks 2 and 3 are about -- which would confound the
              comparator with the object of study. Environmental folds now assign the
              Koppen group x GLiM lithology strata WHOLE to folds, so no stratum
              straddles the train/test boundary. The old column survives, renamed
              fold_predictor_kmeans, as a released sensitivity and named for what it is.
  task2p      new. Task 2 has no LOSO estimate (single-class test fold), so the
              held-out provenance split is pairwise: within each of the two surveys of
              a pair, sites are split 80/20, and the classifier is trained on both
              surveys' training sites and tested on both surveys' held-out sites.
              Two classes in the test set, chance 1/2, no site on both sides.
  nndm        the geographic target distribution is reported per survey AND across
              surveys, so the vacuity is in the released table and not only in prose.

RUN (PowerShell):
  python "D:\claude_projects\Geochemistry\01-paper2\05-prereg\measure_autocorrelation_range.py"
  python "D:\claude_projects\Geochemistry\01-paper2\05-prereg\build_fold_definitions_v2.py"
"""
import io, os, sys, itertools
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEED = 20260925
INT = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
OUT = r"D:\claude_projects\Geochemistry\01-paper2\05-prereg"
CW = r"D:\claude_projects\Geochemistry\01-paper2\03-crosswalk"
K = 10
FIVE = ["Li", "In", "Te", "Tl", "Se"]
TEST_FRAC = 0.20
SITE_DP = 5              # coordinate rounding that defines a site
R_EARTH = 6371.0088


def greedy_balance(sizes, k, rng):
    """Assign groups to k folds, largest first, always into the lightest fold.

    Deterministic given the seed. Ties in size are broken by a seeded shuffle before
    sorting, so the result does not depend on the input ordering of equal groups.
    """
    keys = list(sizes)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -sizes[g])
    load = np.zeros(k, dtype=np.int64)
    out = {}
    for g in keys:
        f = int(np.argmin(load))
        out[g] = f
        load[f] += sizes[g]
    return out, load


def nearest_other_survey_km(lat, lon, survey):
    """Median distance from each survey's samples to the nearest sample of a
    DIFFERENT survey, and the within-survey nearest-neighbour median for contrast.

    This is the measurement that makes NNDM-geographic vacuous across surveys: NNDM
    deletes training points until the training-to-test distance distribution matches
    the target. If the nearest point of another survey is thousands of km away, the
    match requires deleting the training set.
    """
    la, lo = np.radians(lat), np.radians(lon)
    rows = []
    for s in pd.unique(survey):
        a = survey == s
        b = ~a
        for label, other in (("other_survey", b), ("within_survey", a)):
            d_min = np.full(int(a.sum()), np.inf)
            idx_o = np.flatnonzero(other)
            # chunk the cross product to keep memory bounded
            for lo_chunk in np.array_split(idx_o, max(1, len(idx_o) // 2000)):
                h = (np.sin((la[a][:, None] - la[lo_chunk][None, :]) / 2) ** 2
                     + np.cos(la[a])[:, None] * np.cos(la[lo_chunk])[None, :]
                     * np.sin((lo[a][:, None] - lo[lo_chunk][None, :]) / 2) ** 2)
                np.clip(h, 0.0, 1.0, out=h)
                d = 2.0 * R_EARTH * np.arcsin(np.sqrt(h))
                if label == "within_survey":
                    # Exclude each point's distance to itself. The mask must be built
                    # against THIS CHUNK's indices, not the whole other-index array,
                    # or the shapes disagree the moment chunking kicks in.
                    same = (lo_chunk[None, :] == np.flatnonzero(a)[:, None])
                    d[same] = np.inf
                d_min = np.minimum(d_min, d.min(axis=1))
            finite = d_min[np.isfinite(d_min)]
            rows.append(dict(survey=s, target=label, n=int(finite.size),
                             median_km=float(np.median(finite)) if finite.size else np.nan,
                             p05_km=float(np.percentile(finite, 5)) if finite.size else np.nan,
                             p95_km=float(np.percentile(finite, 95)) if finite.size else np.nan))
    return pd.DataFrame(rows)


def main():
    bs_path = os.path.join(OUT, "autocorrelation-block-size.csv")
    if not os.path.exists(bs_path):
        print("STOP. autocorrelation-block-size.csv is missing.")
        print("Run measure_autocorrelation_range.py first. The spatial block side is")
        print("a measurement, not a default, so this script will not guess one.")
        sys.exit(2)
    bs = pd.read_csv(bs_path).iloc[0]
    B = float(bs.block_side_km)
    print(f"block side b = {B:.0f} km  (p{int(bs.percentile)} of "
          f"{int(bs.n_accepted)} accepted variogram ranges)")

    frames = []
    for s in ["gemas", "ngsa", "nasgl"]:
        m = pd.read_parquet(os.path.join(INT, f"{s}_meta.parquet"))[
            ["survey", "sample_id", "latitude", "longitude"]].copy()
        c = pd.read_parquet(os.path.join(INT, f"{s}_conc.parquet"))
        m["survey"] = s.upper()
        frames.append((m.reset_index(drop=True), c.reset_index(drop=True)))
    core = sorted(set.intersection(*[set(c.columns) for _, c in frames]) - set(FIVE))
    print(f"primary core elements: {len(core)}")

    meta = pd.concat([m for m, _ in frames], ignore_index=True)
    conc = pd.concat([c[core] for _, c in frames], ignore_index=True)
    meta["row_id"] = np.arange(len(meta))
    lat = pd.to_numeric(meta.latitude, errors="coerce")
    lon = pd.to_numeric(meta.longitude, errors="coerce")
    has_xy = lat.notna() & lon.notna()
    print(f"rows: {len(meta):,}   with coordinates: {int(has_xy.sum()):,}")

    rng = np.random.default_rng(SEED)

    # --- random k-fold, stratified by survey -------------------------------------
    f = np.full(len(meta), -1, dtype=int)
    for s, idx in meta.groupby("survey").groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        f[idx] = np.arange(len(idx)) % K
    meta["fold_random_kfold"] = f

    # --- LOSO --------------------------------------------------------------------
    meta["fold_loso"] = meta["survey"].map({"GEMAS": 0, "NGSA": 1, "NASGL": 2})

    # --- spatial blocking at the measured side ------------------------------------
    # sinusoidal metres-per-degree: adequate for gridding, not for distance claims
    x_km = lon * 111.320 * np.cos(np.radians(lat))
    y_km = lat * 110.574
    bx = np.floor(x_km / B)
    by = np.floor(y_km / B)
    cell = (bx.astype("Int64").astype(str) + "_" + by.astype("Int64").astype(str))
    cell = cell.where(has_xy)
    meta["spatial_block"] = cell

    # A block belongs to exactly one survey, because the surveys are geographically
    # disjoint. So the number of folds spatial blocking can support is capped by the
    # SMALLEST per-survey block count: with fewer blocks than folds in one survey,
    # some folds contain no sample from that survey -- which reintroduces exactly the
    # single-missing-class defect that removed Task 2 from LOSO, this time inside the
    # spatial protocol. Assigning blocks to folds globally hides this; assigning them
    # WITHIN each survey exposes it and then guarantees every fold carries all three.
    per_survey_blocks = {}
    for s in meta.survey.unique():
        per_survey_blocks[s] = int(
            cell[(meta.survey == s) & has_xy].dropna().nunique())
    print(f"spatial blocks at {B:.0f} km: {int(cell.dropna().nunique())} total")
    for s, n in per_survey_blocks.items():
        print(f"    {s:6s} {n:4d} blocks")
    # K for spatial blocking is DERIVED, by a rule fixed before the number was known:
    # every spatial fold must draw on at least MIN_BLOCKS_PER_FOLD blocks from EVERY
    # survey, so that no fold's estimate rests on one block's idiosyncrasy. With one
    # block per fold, fold size equals block size, and block populations here span
    # 3 to 1016 samples -- a fold holding 3 samples of a survey is not an estimate.
    #   K_spatial = min over surveys of floor(blocks / MIN_BLOCKS_PER_FOLD), capped at K
    MIN_BLOCKS_PER_FOLD = 3
    min_blocks = min(per_survey_blocks.values())
    thinnest = min(per_survey_blocks, key=per_survey_blocks.get)
    K_SP = max(1, min(K, min_blocks // MIN_BLOCKS_PER_FOLD))
    print(f"  K_spatial rule: >= {MIN_BLOCKS_PER_FOLD} blocks per survey per fold.")
    print(f"  {thinnest} is thinnest at {min_blocks} blocks -> K_spatial = {K_SP}"
          + (f" (capped down from {K})" if K_SP < K else ""))
    if K_SP < K:
        print(f"  At b = {B:.0f} km these continental surveys support only "
              f"{min_blocks}-{max(per_survey_blocks.values())} blocks each, so a "
              f"{K}-fold")
        print("  spatially-blocked CV is arithmetically impossible, not merely "
              "awkward. This is")
        print("  a consequence of the measured range, not a tuning choice.")
    if K_SP <= 3:
        print(f"  NOTE: spatial blocking now has {K_SP} folds, the same as LOSO. That "
              "is good for")
        print("  the comparison -- matched fold count, so the only difference is the "
              "blocking axis --")
        print("  but section 3.5's 'no significance claim on three folds' now binds "
              "spatial")
        print("  blocking too, not only LOSO. The methodology must say so.")
    if K_SP < 2:
        print("  *** STOP. Fewer than 2 spatial folds. Spatial blocking cannot serve "
              "as a")
        print("      comparator at this block size and must be reported as "
              "unavailable, not faked.")
    fold_sp = pd.Series(pd.NA, index=meta.index, dtype="Int64")
    for i, s in enumerate(sorted(per_survey_blocks)):
        sel = (meta.survey == s) & has_xy & cell.notna()
        sizes_s = cell[sel].value_counts().to_dict()
        assign_s, load_s = greedy_balance(sizes_s, K_SP,
                                          np.random.default_rng(SEED + 1 + i))
        fold_sp.loc[sel] = cell[sel].map(assign_s).astype("Int64")
        print(f"    {s:6s} -> {K_SP} folds, sizes {load_s.min()}-{load_s.max()}")
    meta["fold_spatial_block"] = fold_sp

    # --- environmental blocking on Koppen x GLiM strata ---------------------------
    # The strata file is NOT row-aligned with meta: build_environmental_strata.py
    # dropped the samples without coordinates, so it has 8,280 rows against meta's
    # 8,285. Positional alignment would silently shift every stratum after the first
    # dropped row onto the wrong sample -- a wrong answer that runs clean. Join on
    # survey plus coordinates instead, and refuse to continue if the join is not
    # one-to-one.
    st = pd.read_csv(os.path.join(CW, "sample-environmental-strata.csv"))
    print(f"\nstrata file rows {len(st):,} vs meta rows {len(meta):,} "
          f"(difference = samples without coordinates: {int((~has_xy).sum())})")

    def coord_key(frame, latcol="latitude", loncol="longitude"):
        return (frame["survey"].astype(str) + "@"
                + pd.to_numeric(frame[latcol], errors="coerce").round(6).astype(str)
                + "," + pd.to_numeric(frame[loncol], errors="coerce").round(6).astype(str))

    st_key = coord_key(st)
    meta_key = coord_key(meta)
    # Duplicate keys are co-located samples in the same survey, which is legitimate:
    # two samples at one site share a stratum. It only breaks the join if the
    # duplicated rows DISAGREE on the stratum -- so check that, rather than refusing
    # outright or collapsing silently.
    dup_mask = st_key.duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    conflicting = []
    if n_dup:
        g = st[dup_mask].assign(_k=st_key[dup_mask])
        for k, grp in g.groupby("_k"):
            if grp[["kg_group", "glim_class", "glim_code"]].drop_duplicates().shape[0] > 1:
                conflicting.append(k)
        print(f"  {n_dup} rows share a survey+coordinate key "
              f"({st_key[dup_mask].nunique()} co-located sites); "
              f"{len(conflicting)} disagree on the stratum")
        if conflicting:
            print("  Co-located samples assigned DIFFERENT strata cannot both be "
                  "right, so those")
            print("  samples are marked unusable for environmental blocking rather "
                  "than guessed at.")
    st_idx = pd.Series(np.arange(len(st)), index=st_key)
    st_idx = st_idx[~st_idx.index.duplicated(keep="first")]
    conflict_keys = set(conflicting)
    pos = meta_key.map(st_idx)
    matched = pos.notna()
    print(f"strata joined on survey+coordinates: {int(matched.sum()):,} of "
          f"{len(meta):,} samples matched, {int((~matched).sum())} unmatched")
    if int((~matched).sum()) > int((~has_xy).sum()):
        print("STOP. more samples are unmatched than lack coordinates, so the join is")
        print("losing real samples rather than only the coordinate-free ones.")
        sys.exit(3)
    take = pos[matched].astype(int).to_numpy()
    kg = pd.Series(pd.NA, index=meta.index, dtype="object")
    gl = pd.Series(pd.NA, index=meta.index, dtype="object")
    gc = pd.Series(np.nan, index=meta.index, dtype="float64")
    kg.loc[matched] = st.kg_group.to_numpy()[take]
    gl.loc[matched] = st.glim_class.to_numpy()[take]
    gc.loc[matched] = pd.to_numeric(st.glim_code, errors="coerce").to_numpy()[take]

    stratum = kg.astype(str) + "|" + gl.astype(str)
    bad = ((~matched) | (kg.astype(str) == "?") | (gc <= 0) | (gc >= 14) | gc.isna()
           | meta_key.isin(conflict_keys))
    stratum = stratum.where(~bad)
    ssizes = stratum.dropna().value_counts().to_dict()
    sassign, sload = greedy_balance(ssizes, K, np.random.default_rng(SEED + 2))
    meta["env_stratum"] = stratum
    meta["fold_environmental"] = stratum.map(sassign).astype("Int64")
    print(f"environmental strata: {len(ssizes)}   "
          f"unusable samples {int(bad.sum())}   fold sizes {sload.min()}-{sload.max()}")

    # --- matched-K baselines ------------------------------------------------------
    # Spatial blocking is forced to K_SP folds by the measured block size, while the
    # other protocols run at K = 10. Comparing a 10-fold estimate against a K_SP-fold
    # estimate confounds the BLOCKING AXIS with TRAINING-SET SIZE: at K = 10 each
    # model sees 90% of the data, at K = 3 only 67%. A protocol comparison that does
    # not hold training size fixed is not a protocol comparison. So every non-LOSO
    # protocol also gets a K_SP-fold version, and the HEADLINE comparison is run at
    # K_SP, matched to both spatial blocking and LOSO's 3 folds.
    if K_SP != K:
        f3 = np.full(len(meta), -1, dtype=int)
        r3 = np.random.default_rng(SEED + 3)
        for s, idx in meta.groupby("survey").groups.items():
            idx = np.array(idx)
            r3.shuffle(idx)
            f3[idx] = np.arange(len(idx)) % K_SP
        meta[f"fold_random_k{K_SP}"] = f3
        sassign3, sload3 = greedy_balance(ssizes, K_SP, np.random.default_rng(SEED + 4))
        meta[f"fold_environmental_k{K_SP}"] = stratum.map(sassign3).astype("Int64")
        print(f"matched-K baselines added at K = {K_SP}: "
              f"fold_random_k{K_SP}, fold_environmental_k{K_SP}")
        print(f"  Headline protocol comparison runs at K = {K_SP} for every protocol "
              f"and for LOSO.")
        print(f"  The K = {K} versions are reported as a secondary, higher-resolution "
              "read of the")
        print("  unblocked and environmentally blocked protocols only, never against "
              "spatial or LOSO.")

    # --- superseded k-means column, kept and renamed honestly ---------------------
    X = conc.apply(pd.to_numeric, errors="coerce")
    X = np.log10(X.clip(lower=1e-9))
    X = X.fillna(X.median())
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X.values)
    meta["fold_predictor_kmeans"] = km.labels_
    print("fold_predictor_kmeans: k-means on log concentrations "
          "(SENSITIVITY ONLY, not the environmental protocol)")

    # --- Task 2p: pairwise provenance, site-level 80/20 --------------------------
    site = (lat.round(SITE_DP).astype(str) + "_" + lon.round(SITE_DP).astype(str))
    site = site.where(has_xy)
    meta["site_key"] = site
    print(f"distinct sites: {site.nunique():,} over {int(has_xy.sum()):,} samples")
    t2p = meta[["row_id", "survey", "sample_id", "site_key"]].copy()
    for a, b in itertools.combinations(["GEMAS", "NGSA", "NASGL"], 2):
        col = f"split_{a}_{b}"
        t2p[col] = pd.Series(pd.NA, index=t2p.index, dtype="string")
        for s in (a, b):
            sites = pd.Index(sorted(site[(meta.survey == s) & has_xy].dropna().unique()))
            r = np.random.default_rng(SEED + abs(hash((a, b, s))) % 10_000)
            pick = r.permutation(len(sites))
            n_test = int(round(TEST_FRAC * len(sites)))
            test_sites = set(sites[pick[:n_test]])
            sel = (meta.survey == s) & has_xy
            t2p.loc[sel, col] = np.where(
                site[sel].isin(test_sites), "test", "train")
        n_tr = int((t2p[col] == "train").sum())
        n_te = int((t2p[col] == "test").sum())
        cls_te = t2p.loc[t2p[col] == "test", "survey"].nunique()
        print(f"  {col}: train {n_tr:5d}  test {n_te:5d}  classes in test {cls_te}"
              + ("  OK" if cls_te == 2 else "  *** BROKEN ***"))

    # --- NNDM target distributions, including the cross-survey vacuity ----------
    # This is a diagnostic, not a fold definition. A failure here must not cost the
    # fold files, which are the actual deliverable -- the first run crashed at exactly
    # this point and wrote nothing.
    try:
        G = nearest_other_survey_km(lat[has_xy].to_numpy(float),
                                    lon[has_xy].to_numpy(float),
                                    meta.survey[has_xy].to_numpy())
    except Exception as e:                                        # noqa: BLE001
        print(f"\nNNDM distance diagnostic FAILED ({type(e).__name__}: {e})")
        print("Fold files are still written below. Rerun once this is fixed.")
        G = None
    print("\nNNDM geographic target distributions (median km):")
    print(G.to_string(index=False) if G is not None else "  unavailable")
    print("A cross-survey median in the thousands of km is why NNDM-geographic is")
    print("reported within survey only (methodology v7 section 3.3).")

    # --- write --------------------------------------------------------------------
    cols = ["row_id", "survey", "sample_id", "latitude", "longitude"]
    rk = ["fold_random_kfold"] + ([f"fold_random_k{K_SP}"]
                                 if f"fold_random_k{K_SP}" in meta.columns else [])
    ek = ["env_stratum", "fold_environmental"] + (
        [f"fold_environmental_k{K_SP}"]
        if f"fold_environmental_k{K_SP}" in meta.columns else [])
    meta[cols + rk].to_csv(
        os.path.join(OUT, "folds-random-kfold.csv"), index=False)
    meta[cols + ["fold_loso"]].to_csv(
        os.path.join(OUT, "folds-loso.csv"), index=False)
    meta[cols + ["spatial_block", "fold_spatial_block"]].to_csv(
        os.path.join(OUT, "folds-spatial-block.csv"), index=False)
    meta[cols + ek].to_csv(
        os.path.join(OUT, "folds-environmental.csv"), index=False)
    meta[cols + ["fold_predictor_kmeans"]].to_csv(
        os.path.join(OUT, "folds-predictor-kmeans-sensitivity.csv"), index=False)
    t2p.to_csv(os.path.join(OUT, "folds-task2p.csv"), index=False)
    if G is not None:
        G.to_csv(os.path.join(OUT, "nndm-target-distributions.csv"), index=False)
    pd.Series(core, name="element").to_csv(
        os.path.join(OUT, "primary-core-elements.csv"), index=False)

    # --- Task 2 eligibility gate --------------------------------------------------
    # Task 2 needs all three classes present in EVERY test fold. LOSO fails this by
    # construction. Any other protocol that also fails it is equally unusable for
    # Task 2, and that has to be decided here, from counts, not assumed in prose.
    # Presence of a class is not sufficiency. A fold holding 3 samples of a survey
    # technically contains all three classes, and a balanced accuracy computed on it
    # is noise dressed as a measurement. Require a minimum per-survey population in
    # every fold as well as presence.
    MIN_PER_SURVEY_PER_FOLD = 30
    print("\nfold occupancy and Task 2 eligibility:")
    print(f"  criteria: every fold contains all 3 surveys AND at least "
          f"{MIN_PER_SURVEY_PER_FOLD} samples of each")
    cand = [c for c in ["fold_random_kfold", f"fold_random_k{K_SP}",
                        "fold_spatial_block", "fold_environmental",
                        f"fold_environmental_k{K_SP}", "fold_predictor_kmeans"]
            if c in meta.columns]
    eligible, ineligible = [], []
    for c in cand:
        t = pd.crosstab(meta[c], meta.survey)
        n_folds = len(t)
        n_all3 = int(t.gt(0).all(axis=1).sum())
        n_enough = int(t.ge(MIN_PER_SURVEY_PER_FOLD).all(axis=1).sum())
        ok = (n_enough == n_folds) and n_folds >= 2
        (eligible if ok else ineligible).append((c, n_folds, n_all3))
        print(f"  {c:28s}: {n_folds:3d} folds, all 3 present in {n_all3:3d}, "
              f"all 3 at n>={MIN_PER_SURVEY_PER_FOLD} in {n_enough:3d}"
              f"   {'ELIGIBLE' if ok else 'NOT ELIGIBLE'}")
        if not ok:
            short = t[~t.ge(MIN_PER_SURVEY_PER_FOLD).all(axis=1)]
            for fold_id, row in short.iterrows():
                thin_s = [f"{s}={int(row[s])}" for s in t.columns
                          if row[s] < MIN_PER_SURVEY_PER_FOLD]
                print(f"      fold {fold_id}: {', '.join(thin_s)}")
    print(f"\n  Task 2 protocol set: "
          f"{', '.join(c for c, _, _ in eligible) if eligible else 'NONE'}")
    if ineligible:
        print("  Excluded from Task 2, for the same single-missing-class reason that")
        print("  excluded LOSO: " + ", ".join(c for c, _, _ in ineligible))
        print("  Methodology section 3.1 must be amended to match this list. Do not")
        print("  report a Task 2 number under an excluded protocol.")
    if not eligible:
        print("  *** Task 2 has NO eligible protocol besides Task 2p. Say so in the")
        print("      paper; do not manufacture one by shrinking K until it passes.")
    print("\n  Task 1 uses every protocol including LOSO. Task 2p is always available,")
    print("  because it holds out sites within a survey pair and so cannot lose a class.")
    print(f"\nwrote 7 files to {OUT}")


if __name__ == "__main__":
    main()
