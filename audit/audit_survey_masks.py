"""Audit the three observational surveys for the Tellus defect class.

Paper 1 encodes censoring differently from Tellus: a censored cell is stored as
conc = NaN with the limit recorded in the LOD matrix, rather than substituted at
half the limit. So the Tellus failure (substituted value not exactly recoverable)
cannot arise. The checks below verify that encoding rather than assume it.
"""
import io, sys, os
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INT = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
OUT = r"D:\claude_projects\Geochemistry\01-paper2\03-crosswalk"
rows = []

for s in ["gemas", "ngsa", "nasgl"]:
    conc = pd.read_parquet(os.path.join(INT, f"{s}_conc.parquet"))
    mask = pd.read_parquet(os.path.join(INT, f"{s}_mask.parquet"))
    lod  = pd.read_parquet(os.path.join(INT, f"{s}_lod.parquet"))
    els = [c for c in conc.columns if c in mask.columns and c in lod.columns]

    M = mask[els].to_numpy() == 1
    V = conc[els].to_numpy(dtype=float)
    L = lod[els].to_numpy(dtype=float)
    vnan, lok = ~np.isfinite(V), np.isfinite(L)

    n_cells   = M.size
    n_flag    = int(M.sum())
    flag_nolod= int((M & ~lok).sum())          # censored with no recorded limit
    lod_noflag= int((lok & ~M).sum())          # limit recorded but not flagged
    flag_value= int((M & ~vnan).sum())         # censored yet a value survives
    val_below = int((~M & np.isfinite(V) & lok & (V < L * (1 - 1e-9))).sum())
    half_sub  = int((np.isfinite(V) & lok & np.isclose(V, 0.5 * L, rtol=1e-6)).sum())

    print(f"\n===== {s.upper()} =====  {conc.shape[0]:,} x {len(els)} = {n_cells:,} cells")
    print(f"  censored cells (mask == 1)                    : {n_flag:,}"
          f"  ({100*n_flag/n_cells:.2f}%)")
    print(f"  censored WITHOUT a recorded limit             : {flag_nolod:,}")
    print(f"  limit recorded but NOT flagged                : {lod_noflag:,}")
    print(f"  censored yet carrying a substituted value     : {flag_value:,}")
    print(f"  uncensored values lying strictly below a limit: {val_below:,}   <- Tellus defect class")
    print(f"  values equal to 0.5 x limit (half-substitution): {half_sub:,}")
    verdict = "CLEAN" if (flag_value == 0 and val_below == 0) else "DEFECT PRESENT"
    print(f"  verdict: {verdict}")
    rows.append(dict(survey=s, cells=n_cells, censored=n_flag,
                     censored_no_limit=flag_nolod, limit_not_flagged=lod_noflag,
                     censored_with_value=flag_value, below_limit_unflagged=val_below,
                     half_substituted=half_sub, verdict=verdict))

r = pd.DataFrame(rows)
r.to_csv(os.path.join(OUT, "survey-mask-audit.csv"), index=False)
print("\n" + r.to_string(index=False))
print("\nwrote", os.path.join(OUT, "survey-mask-audit.csv"))
