"""KILL TEST for methodology section 3.4.1.

The stratified label permutation has power only if strata contain more than one
survey. Geographic strata cannot: the three observational surveys are disjoint.
This script builds environmental strata (Koppen climate x GLiM lithology) and
reports how many are multi-survey, and how much permutation entropy remains.

Inputs
  Koppen : Beck et al. 2018, Sci Data, 10.1038/sdata.2018.214 (1 km GeoTIFF)
  GLiM   : Hartmann & Moosdorf 2012, 10.1594/PANGAEA.788537 (0.5 deg ASCII)
"""
import io, sys, os, math
import numpy as np, pandas as pd
import rasterio
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INT = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
COV = r"D:\claude_projects\Geochemistry\04-data\COVARIATES"
OUT = r"D:\claude_projects\Geochemistry\01-paper2\03-crosswalk"
KG  = os.path.join(COV, "koppen", "Beck_KG_V1_present_0p0083.tif")
GL  = os.path.join(COV, "glim",  "glim_wgs84_0point5deg.txt.asc")

# ---------- points
pts = []
for s in ["gemas", "ngsa", "nasgl"]:
    m = pd.read_parquet(os.path.join(INT, f"{s}_meta.parquet"))
    m = m[["survey", "sample_id", "latitude", "longitude"]].copy()
    m["survey"] = s.upper()
    pts.append(m)
P = pd.concat(pts, ignore_index=True).dropna(subset=["latitude", "longitude"])
print("points with coordinates:", len(P))
print(P.groupby("survey").size().to_string(), "\n")

# ---------- Koppen legend
leg = {}
with open(os.path.join(COV, "koppen", "legend.txt"), encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.split(":")
        if len(parts) >= 2 and parts[0].strip().isdigit():
            leg[int(parts[0].strip())] = parts[1].split("[")[0].strip()
print("Koppen legend entries:", len(leg), "| e.g.", list(leg.items())[:4], "\n")

# ---------- sample Koppen
with rasterio.open(KG) as src:
    print("Koppen raster:", src.width, "x", src.height, "| crs", src.crs, "| nodata", src.nodata)
    vals = np.array([v[0] for v in src.sample(
        [(float(x), float(y)) for x, y in zip(P.longitude, P.latitude)])])
P["kg_code"] = vals
P["kg_class"] = [leg.get(int(v), f"code{int(v)}") for v in vals]
P["kg_group"] = [c[0] if c and c[0] in "ABCDE" else "?" for c in P["kg_class"]]

# ---------- sample GLiM (ESRI ASCII grid)
hdr = {}
with open(GL, encoding="utf-8", errors="replace") as f:
    for _ in range(6):
        k, v = f.readline().split()
        hdr[k.lower()] = float(v)
    grid = np.loadtxt(f)
print("GLiM grid:", grid.shape, "| header", {k: hdr[k] for k in
      ("ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value")})

cs, x0, y0 = hdr["cellsize"], hdr["xllcorner"], hdr["yllcorner"]
nr, nc = int(hdr["nrows"]), int(hdr["ncols"])
col = np.floor((P.longitude.to_numpy() - x0) / cs).astype(int)
row = (nr - 1 - np.floor((P.latitude.to_numpy() - y0) / cs)).astype(int)
ok = (col >= 0) & (col < nc) & (row >= 0) & (row < nr)
lith = np.full(len(P), hdr["nodata_value"])
lith[ok] = grid[row[ok], col[ok]]
P["glim_code"] = lith.astype(int)

GLIM_NAMES = {1: "su_sedimentary_undiff", 2: "ss_siliciclastic", 3: "sm_mixed_sed",
              4: "sc_carbonate", 5: "ev_evaporite", 6: "mt_metamorphic",
              7: "pa_acid_plutonic", 8: "pi_intermediate_plutonic",
              9: "pb_basic_plutonic", 10: "va_acid_volcanic",
              11: "vi_intermediate_volcanic", 12: "vb_basic_volcanic",
              13: "ig_ice_glacier", 14: "wb_water_bodies", 15: "nd_no_data"}
P["glim_class"] = [GLIM_NAMES.get(c, f"code{c}") for c in P["glim_code"]]

miss = int(((P.kg_group == "?") | (P.glim_code <= 0) | (P.glim_code >= 14)).sum())
print(f"\npoints missing a usable climate or lithology class: {miss} "
      f"({100*miss/len(P):.1f}%)  -> excluded from strata\n")
V = P[(P.kg_group != "?") & (P.glim_code > 0) & (P.glim_code < 14)].copy()

# ---------- strata definitions, coarse to fine
DEFS = {
    "climate group only (A-E)":                 ["kg_group"],
    "lithology only (GLiM)":                    ["glim_class"],
    "climate group x lithology":                ["kg_group", "glim_class"],
    "full Koppen class x lithology":            ["kg_class", "glim_class"],
}

summary = []
for name, keys in DEFS.items():
    g = V.groupby(keys)["survey"].agg(n="size", k=lambda s: s.nunique())
    multi = g[g.k >= 2]
    tab = V.groupby(keys + ["survey"]).size().unstack("survey", fill_value=0)
    mt = tab.loc[multi.index] if len(multi) else tab.iloc[0:0]
    # permutation entropy: log of the number of distinct within-stratum label
    # arrangements, summed over multi-survey strata (nats)
    ent = 0.0
    for _, r in mt.iterrows():
        n = int(r.sum())
        ent += math.lgamma(n + 1) - sum(math.lgamma(int(c) + 1) for c in r if c > 0)
    summary.append(dict(definition=name, strata=len(g), multi_survey_strata=len(multi),
                        samples_in_multi=int(multi.n.sum()),
                        pct_samples_in_multi=round(100*int(multi.n.sum())/len(V), 1),
                        surveys_covered=int(mt.gt(0).any(axis=0).sum()) if len(mt) else 0,
                        min_survey_n_in_multi=int(mt.sum().min()) if len(mt) else 0,
                        log_permutations=round(ent, 1)))
    print("=" * 74)
    print(name)
    print(f"  strata                        : {len(g)}")
    print(f"  strata with >=2 surveys       : {len(multi)}")
    print(f"  samples in those strata       : {int(multi.n.sum()):,} of {len(V):,}"
          f"  ({100*int(multi.n.sum())/len(V):.1f}%)")
    if len(mt):
        print(f"  per-survey n inside them      : {mt.sum().to_dict()}")
        print(f"  log(distinct permutations)    : {ent:,.1f} nats")
        print("  the multi-survey strata:")
        print(mt.head(25).to_string())
    else:
        print("  -> permutation reduces to the identity. NO POWER.")
    print()

S = pd.DataFrame(summary)
P.to_csv(os.path.join(OUT, "sample-environmental-strata.csv"), index=False)
S.to_csv(os.path.join(OUT, "strata-power-summary.csv"), index=False)
print("=" * 74)
print(S.to_string(index=False))
print("\nwrote sample-environmental-strata.csv and strata-power-summary.csv to", OUT)
