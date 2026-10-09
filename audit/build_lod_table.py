"""Stage 2, step 1: per-element detection-limit table for all four sources, and
the cross-survey disparity Delta used by Task 3.

Checks before any ratio is reported:
  A  is the limit constant per element within a survey, or per-sample?
  B  is the limit in the same unit as the concentration column? (magnitude test
     against the smallest uncensored value)
  C  are there elements where a limit is unavailable, so Delta is undefined?
"""
import io, sys, os
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 220)

INT = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
OUT = r"D:\claude_projects\Geochemistry\01-paper2\03-crosswalk"
XL  = r"D:\claude_projects\Geochemistry\04-data\TELLUS\topsoil_A\58xxxxA-65xxxxA_Shallow_Topsoil_Download_v1.2.xlsx"
FIVE_SET_ASIDE = ["Li", "In", "Te", "Tl", "Se"]

rows = []
for s in ["gemas", "ngsa", "nasgl"]:
    conc = pd.read_parquet(os.path.join(INT, f"{s}_conc.parquet"))
    lod  = pd.read_parquet(os.path.join(INT, f"{s}_lod.parquet"))
    mask = pd.read_parquet(os.path.join(INT, f"{s}_mask.parquet"))
    for e in conc.columns:
        if e not in lod.columns:
            continue
        L = pd.to_numeric(lod[e], errors="coerce").dropna()
        v = pd.to_numeric(conc[e], errors="coerce")
        m = mask[e].fillna(0).to_numpy(dtype=bool) if e in mask.columns else np.zeros(len(v), bool)
        uncens_min = float(np.nanmin(v[~m])) if (~m).sum() and v[~m].notna().any() else np.nan
        rows.append(dict(
            survey=s.upper(), element=e,
            n_limit_cells=int(len(L)), n_distinct=int(L.nunique()),
            lod_min=float(L.min()) if len(L) else np.nan,
            lod_median=float(L.median()) if len(L) else np.nan,
            lod_max=float(L.max()) if len(L) else np.nan,
            constant=bool(len(L) and L.nunique() == 1),
            n_censored=int(m.sum()),
            min_uncensored=uncens_min))

# ---- Tellus, accredited limits, both routes
u = pd.read_excel(XL, sheet_name="Units_Detection_Limits")
u.columns = [str(c).strip() for c in u.columns]
u["Analyte"] = u["Analyte"].astype(str).str.strip()
u["Accredited"] = u["Accredited"].astype(str).str.strip().str.lower()
u["LLD"] = pd.to_numeric(u["LLD"], errors="coerce")
u = u[u["Analyte"].str.contains("_", na=False)]
OX = {"SiO2": "Si", "TiO2": "Ti", "Al2O3": "Al", "Fe2O3": "Fe", "MnO": "Mn",
      "MgO": "Mg", "CaO": "Ca", "Na2O": "Na", "K2O": "K", "P2O5": "P", "SO3": "S"}
for a, g in u.groupby("Analyte", sort=False):
    acc = g[g["Accredited"] == "yes"]
    r = acc.iloc[0] if len(acc) else g.iloc[0]
    if not np.isfinite(r["LLD"]):
        continue
    route = "TELLUS_XRFS" if a.endswith("XRFS") else ("TELLUS_ICPar" if a.endswith("ICPar") else None)
    if route is None:
        continue
    stem = a.split("_")[0]
    el = OX.get(stem, stem)
    unit = "%" if "_%" in a else ("mg/kg" if "_mgkg" in a else ("ug/kg" if "gkg" in a else "?"))
    lod_mgkg = r["LLD"] * 1e4 if unit == "%" else (r["LLD"] / 1000 if unit == "ug/kg" else r["LLD"])
    rows.append(dict(survey=route, element=el, n_limit_cells=1, n_distinct=1,
                     lod_min=lod_mgkg, lod_median=lod_mgkg, lod_max=lod_mgkg,
                     constant=True, n_censored=np.nan, min_uncensored=np.nan,
                     raw_unit=unit, raw_lld=r["LLD"],
                     accredited=(len(acc) > 0)))

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "lod-table-per-element-survey.csv"), index=False)

print("############ A. is the limit constant per element within a survey?")
print(T.groupby("survey")["constant"].agg(["sum", "count"]).to_string())
nonconst = T[(~T.constant) & (T.n_limit_cells > 0)]
print("non-constant element-survey pairs:", len(nonconst))
if len(nonconst):
    print(nonconst[["survey", "element", "n_distinct", "lod_min", "lod_max"]].head(20).to_string(index=False))
print()

print("############ B. unit sanity: limit must not exceed the smallest uncensored value")
bad = T[(T.min_uncensored.notna()) & (T.lod_median.notna()) &
        (T.lod_median > T.min_uncensored * 1.0001)]
print("element-survey pairs where median LOD > min uncensored value:", len(bad))
if len(bad):
    print(bad[["survey", "element", "lod_median", "min_uncensored"]].head(20).to_string(index=False))
print()

print("############ C. limit availability across the three observational surveys")
obs = T[T.survey.isin(["GEMAS", "NGSA", "NASGL"])]
piv = obs.pivot_table(index="element", columns="survey", values="lod_median")
have = piv.notna().sum(axis=1)
print("elements with a limit in all three :", int((have == 3).sum()))
print("elements with a limit in two       :", int((have == 2).sum()))
print("elements with a limit in one       :", int((have == 1).sum()))
print("elements with none                 :", int((have == 0).sum()))
print()

# ---- Delta
d = piv.copy()
d["GEMAS_over_NGSA"]  = d.get("GEMAS") / d.get("NGSA")
d["GEMAS_over_NASGL"] = d.get("GEMAS") / d.get("NASGL")
d["NGSA_over_NASGL"]  = d.get("NGSA")  / d.get("NASGL")
rat = d[["GEMAS_over_NGSA", "GEMAS_over_NASGL", "NGSA_over_NASGL"]]
fold = rat.apply(lambda c: np.maximum(c, 1 / c))
d["delta_max_fold"] = fold.max(axis=1)
d["n_pairs"] = rat.notna().sum(axis=1)
d = d.sort_values("delta_max_fold", ascending=False)
d.to_csv(os.path.join(OUT, "lod-disparity-delta.csv"))

print("############ Delta: largest fold difference in detection limit per element")
full = d[d.n_pairs == 3]
print(f"elements with all three pairwise ratios : {len(full)}")
for th in [2, 5, 10, 100, 1000]:
    print(f"  Delta >= {th:>4}x : {int((full.delta_max_fold >= th).sum()):>3} of {len(full)}")
print(f"  median Delta       : {full.delta_max_fold.median():.2f}x")
print()
print("top 20:")
print(d.head(20)[["GEMAS", "NGSA", "NASGL", "delta_max_fold", "n_pairs"]].to_string())
print()
print("the five elements set aside in 2.1:")
print(d.loc[[e for e in FIVE_SET_ASIDE if e in d.index],
            ["GEMAS", "NGSA", "NASGL", "delta_max_fold"]].to_string())
print("\nwrote lod-table-per-element-survey.csv and lod-disparity-delta.csv to", OUT)
