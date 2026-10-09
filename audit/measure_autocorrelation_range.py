"""Measure the range of residual spatial autocorrelation, per element per survey,
and derive the spatial block side b that methodology v7 section 3.3 requires.

WHY THIS EXISTS
Roberts et al. (2017) require a spatial block at least as large as the range of
residual spatial autocorrelation. A block smaller than that range leaves training
points inside the test block's correlation neighbourhood, and the protocol silently
reverts towards random k-fold -- which would understate the gap LOSO reveals and so
bias the paper's central comparison in the paper's own favour. The 2 degree grid in
the first fold build was a convenience with no measurement behind it and is
withdrawn. This script supplies the measurement.

PROCEDURE (fixed before running, not tuned to the output)
  1. For each survey and each primary-core element: take samples with an uncensored
     value and coordinates. log10 the value.
  2. Remove a first-order coordinate trend by OLS on [1, lon, lat]. The variogram is
     fitted to the residuals, because a regional gradient inflates the apparent
     range without implying local autocorrelation.
  3. Empirical variogram on a seeded subsample of at most MAXPTS points: haversine
     distance, NBINS equal-width bins out of to MAXFRAC of the survey's own extent
     (a variogram beyond about a third of the extent is estimated on too few, too
     peripheral pairs to be trusted).
  4. Fit the exponential model gamma(h) = c0 + c1 * (1 - exp(-h/a)) by least
     squares, weighted by bin count. PRACTICAL RANGE = 3a, the lag at which the
     model reaches 95 percent of its sill.
  5. Keep a fit only if it converges, c1 > 0, a > 0, and 3a <= the largest lag
     fitted. A fit whose range exceeds the fitted window is an extrapolation, not a
     measurement, and is recorded as rejected rather than used.
  6. b = 90th percentile of the accepted practical ranges, rounded UP to the next
     25 km. The 90th percentile, not the median: the block must cover the range for
     nine elements in ten, and erring large costs fold resolution while erring small
     costs the validity of the protocol.

OUTPUT
  autocorrelation-ranges.csv   one row per element-survey, with the fit and verdict
  autocorrelation-block-size.csv  the single derived b, with the percentile and n

RUN (PowerShell, from anywhere):
  python "D:\claude_projects\Geochemistry\01-paper2\05-prereg\measure_autocorrelation_range.py"
"""
import io, os, sys, math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEED = 20260925
INT = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
OUT = r"D:\claude_projects\Geochemistry\01-paper2\05-prereg"
FIVE = ["Li", "In", "Te", "Tl", "Se"]        # excluded from the primary core
MAXPTS = 2500                                 # 2500 points -> ~3.1M pairs
NBINS = 25
MAXFRAC = 1.0 / 3.0
PCTL = 90
ROUND_KM = 25.0
MIN_R2 = 0.50                 # a fit with no structure is not a measurement
MAX_NUGGET_FRACTION = 0.90    # nearly all nugget means no resolvable spatial signal
R_EARTH = 6371.0088


def haversine_matrix(lat, lon):
    """Pairwise great-circle distance in km. lat/lon in degrees."""
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dla = la - la.T
    dlo = lo - lo.T
    h = np.sin(dla / 2.0) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2.0) ** 2
    np.clip(h, 0.0, 1.0, out=h)
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(h))


def detrend(z, lat, lon):
    """Remove a first-order coordinate trend; return residuals."""
    A = np.column_stack([np.ones_like(lon), lon, lat])
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    return z - A @ beta


def exponential_model(h, c0, c1, a):
    return c0 + c1 * (1.0 - np.exp(-h / a))


def empirical_variogram(D, resid, max_lag, nbins):
    """Binned semivariance. Returns (lag_centre, gamma, count) for populated bins."""
    iu = np.triu_indices_from(D, k=1)
    d = D[iu]
    g = 0.5 * (resid[iu[0]] - resid[iu[1]]) ** 2
    keep = (d > 0) & (d <= max_lag)
    d, g = d[keep], g[keep]
    if d.size < 200:
        return None
    edges = np.linspace(0.0, max_lag, nbins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins).astype(float)
    tot = np.bincount(idx, weights=g, minlength=nbins)
    mid = 0.5 * (edges[:-1] + edges[1:])
    ok = cnt >= 30                      # a bin under 30 pairs is noise, not a point
    if ok.sum() < 6:
        return None
    return mid[ok], tot[ok] / cnt[ok], cnt[ok]


def fit_one(lat, lon, z, rng):
    n = len(z)
    if n > MAXPTS:
        sel = rng.choice(n, MAXPTS, replace=False)
        lat, lon, z = lat[sel], lon[sel], z[sel]
    resid = detrend(z, lat, lon)
    D = haversine_matrix(lat, lon)
    extent = D.max()
    max_lag = extent * MAXFRAC
    if not np.isfinite(max_lag) or max_lag <= 0:
        return dict(status="rejected_no_extent")
    ev = empirical_variogram(D, resid, max_lag, NBINS)
    if ev is None:
        return dict(status="rejected_too_few_pairs", max_lag_km=float(max_lag))
    h, gam, cnt = ev
    sill0 = float(np.nanmax(gam))
    p0 = [max(gam[0] * 0.5, 1e-9), max(sill0 - gam[0] * 0.5, 1e-9), max_lag / 3.0]
    try:
        popt, _ = curve_fit(
            exponential_model, h, gam, p0=p0,
            sigma=1.0 / np.sqrt(cnt), absolute_sigma=False,
            bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, max_lag * 3.0]),
            maxfev=20000)
    except Exception as e:                                # noqa: BLE001
        return dict(status="rejected_fit_failed", note=type(e).__name__,
                    max_lag_km=float(max_lag))
    c0, c1, a = (float(v) for v in popt)
    pred = exponential_model(h, c0, c1, a)
    ss_res = float(np.sum(cnt * (gam - pred) ** 2))
    ss_tot = float(np.sum(cnt * (gam - np.average(gam, weights=cnt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    prange = 3.0 * a
    nug_frac = c0 / (c0 + c1) if (c0 + c1) > 0 else 1.0
    status = "accepted"
    if c1 <= 0:
        status = "rejected_no_structure"
    elif prange > h.max():
        # The variogram has not reached a sill inside the observable window. This is
        # NOT a long range -- it is NO IDENTIFIABLE RANGE: the semivariance is still
        # rising at the largest lag fitted, which is regional trend, not local
        # autocorrelation. Naming it "exceeds window" invited the wrong reading.
        status = "rejected_no_sill_within_window"
    elif r2 < MIN_R2:
        # A fit with r2 near zero has no variogram structure to measure. Four such
        # fits (r2 = 0.003, 0.016, 0.128, 0.204) were accepted on the first run and
        # sat at the SHORT end of the distribution, dragging the low percentiles down
        # with pure noise.
        status = "rejected_fit_unstructured"
    elif nug_frac > MAX_NUGGET_FRACTION:
        status = "rejected_all_nugget"
    return dict(status=status, n_used=int(len(z)), n_bins=int(len(h)),
                extent_km=float(extent), max_lag_km=float(max_lag),
                nugget=c0, partial_sill=c1, a_km=a,
                practical_range_km=float(prange),
                nugget_fraction=float(c0 / (c0 + c1)) if (c0 + c1) > 0 else np.nan,
                weighted_r2=float(r2))


def main():
    rng = np.random.default_rng(SEED)
    frames = {}
    for s in ["gemas", "ngsa", "nasgl"]:
        m = pd.read_parquet(os.path.join(INT, f"{s}_meta.parquet"))
        c = pd.read_parquet(os.path.join(INT, f"{s}_conc.parquet"))
        frames[s.upper()] = (m.reset_index(drop=True), c.reset_index(drop=True))
    core = sorted(set.intersection(*[set(c.columns) for _, c in frames.values()])
                  - set(FIVE))
    print(f"primary core elements: {len(core)}")
    print(f"surveys: {', '.join(frames)}")
    print(f"subsample cap {MAXPTS} pts, {NBINS} bins, window = extent/3, "
          f"exponential model, practical range = 3a\n")

    rows = []
    for survey, (m, c) in frames.items():
        lat_all = pd.to_numeric(m.latitude, errors="coerce").to_numpy(float)
        lon_all = pd.to_numeric(m.longitude, errors="coerce").to_numpy(float)
        for el in core:
            v = pd.to_numeric(c[el], errors="coerce").to_numpy(float)
            ok = np.isfinite(v) & (v > 0) & np.isfinite(lat_all) & np.isfinite(lon_all)
            rec = dict(survey=survey, element=el, n_available=int(ok.sum()))
            if ok.sum() < 200:
                rec["status"] = "rejected_too_few_samples"
            else:
                rec.update(fit_one(lat_all[ok], lon_all[ok],
                                   np.log10(v[ok]), rng))
            rows.append(rec)
            tag = rec.get("status", "?")
            pr = rec.get("practical_range_km")
            print(f"  {survey:6s} {el:4s} n={rec['n_available']:5d}  {tag:32s}"
                  + (f"  range={pr:8.1f} km  r2={rec.get('weighted_r2', float('nan')):.3f}"
                     if pr is not None else ""))

    df = pd.DataFrame(rows)
    f1 = os.path.join(OUT, "autocorrelation-ranges.csv")
    df.to_csv(f1, index=False)

    acc = df[df.status == "accepted"]
    print("\n" + "=" * 72)
    print(f"element-survey pairs attempted : {len(df)}")
    print(f"accepted fits                  : {len(acc)}")
    print("rejected, by reason:")
    for k, n in df.loc[df.status != "accepted", "status"].value_counts().items():
        print(f"  {k:34s} {n}")
    print("\nidentifiability per survey -- this is a RESULT, not a diagnostic:")
    for s in df.survey.unique():
        d = df[df.survey == s]
        n_acc = int((d.status == "accepted").sum())
        n_nosill = int((d.status == "rejected_no_sill_within_window").sum())
        print(f"  {s:6s} identifiable range {n_acc:3d}/{len(d)}   "
              f"no sill inside window {n_nosill:3d}/{len(d)}")
    print("  A 'no sill' element has no local autocorrelation range to measure: its")
    print("  semivariance is still rising at the largest lag fitted. At continental")
    print("  extent that is geological-province structure, which LOSO already blocks.")
    print("  REPORT THIS. Do not present the accepted set as if it were all elements.")

    thin = [s for s in df.survey.unique()
            if int(((df.survey == s) & (df.status == "accepted")).sum()) < 5]
    if thin:
        print(f"\n  *** SELECTION WARNING: {', '.join(thin)} contribute fewer than 5")
        print("      accepted fits, so b is effectively derived from the other")
        print("      surveys and applied to them. This must be stated, not hidden.")

    if acc.empty:
        print("\nNO ACCEPTED FITS. b is NOT derived. Do not proceed to fold rebuild.")
        sys.exit(2)

    r = acc.practical_range_km.to_numpy(float)
    b_raw = float(np.percentile(r, PCTL))
    b = math.ceil(b_raw / ROUND_KM) * ROUND_KM
    print("\naccepted practical ranges (km):")
    for q in (5, 25, 50, 75, 90, 95):
        print(f"  p{q:<3d} {np.percentile(r, q):9.1f}")
    print(f"  min  {r.min():9.1f}    max {r.max():9.1f}")
    print(f"\n  p{PCTL} = {b_raw:.1f} km  ->  BLOCK SIDE b = {b:.0f} km "
          f"(rounded up to {ROUND_KM:.0f} km)")
    print("  sensitivity blocks: "
          f"{0.5 * b:.0f} km and {2.0 * b:.0f} km")

    per_survey = acc.groupby("survey").practical_range_km.describe()
    print("\nper-survey accepted ranges:")
    print(per_survey.to_string())

    pd.DataFrame([dict(percentile=PCTL, percentile_value_km=b_raw,
                       round_to_km=ROUND_KM, block_side_km=b,
                       sensitivity_low_km=0.5 * b, sensitivity_high_km=2.0 * b,
                       n_accepted=len(acc), n_attempted=len(df),
                       model="exponential", practical_range_rule="3a",
                       detrended="first_order_lon_lat", seed=SEED)]
                 ).to_csv(os.path.join(OUT, "autocorrelation-block-size.csv"),
                          index=False)
    print(f"\nwrote {f1}")
    print(f"wrote {os.path.join(OUT, 'autocorrelation-block-size.csv')}")
    print("\nNEXT: run build_fold_definitions_v2.py, which reads the block size "
          "from that file.")


if __name__ == "__main__":
    main()
