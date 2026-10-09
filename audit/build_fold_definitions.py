"""Stage 2, step 2: frozen fold definitions for every GeoShift protocol.

Fold assignment is a DESIGN choice, not a model fit: it uses coordinates, survey
labels and predictor values but never the target of any task, so computing it on the
full table introduces no train/test leakage. All randomness is seeded.
"""
import io, sys, os
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEED = 20260925
INT = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
OUT = r"D:\claude_projects\Geochemistry\01-paper2\05-prereg"
K = 10
FIVE = ["Li", "In", "Te", "Tl", "Se"]

frames = []
for s in ["gemas", "ngsa", "nasgl"]:
    m = pd.read_parquet(os.path.join(INT, f"{s}_meta.parquet"))[
        ["survey", "sample_id", "latitude", "longitude"]].copy()
    c = pd.read_parquet(os.path.join(INT, f"{s}_conc.parquet"))
    k = pd.read_parquet(os.path.join(INT, f"{s}_mask.parquet"))
    m["survey"] = s.upper()
    frames.append((m.reset_index(drop=True), c.reset_index(drop=True), k.reset_index(drop=True)))

core = sorted(set.intersection(*[set(c.columns) for _, c, _ in frames]) - set(FIVE))
print(f"primary core elements: {len(core)}")

meta = pd.concat([m for m, _, _ in frames], ignore_index=True)
conc = pd.concat([c[core] for _, c, _ in frames], ignore_index=True)
mask = pd.concat([k[core] for _, _, k in frames], ignore_index=True)
meta["row_id"] = np.arange(len(meta))
print(f"rows: {len(meta):,}   with coordinates: {int(meta.latitude.notna().sum()):,}")

rng = np.random.default_rng(SEED)

f = np.full(len(meta), -1, dtype=int)
for s, idx in meta.groupby("survey").groups.items():
    idx = np.array(idx); rng.shuffle(idx)
    f[idx] = np.arange(len(idx)) % K
meta["fold_random_kfold"] = f

meta["fold_loso"] = meta["survey"].map({"GEMAS": 0, "NGSA": 1, "NASGL": 2})

lon = pd.to_numeric(meta.longitude, errors="coerce")
lat = pd.to_numeric(meta.latitude, errors="coerce")
cell = np.floor(lon / 2.0).astype("Int64").astype(str) + "_" + np.floor(lat / 2.0).astype("Int64").astype(str)
cell = cell.where(lon.notna() & lat.notna())
meta["spatial_block"] = cell
cells = pd.Index(sorted(cell.dropna().unique()))
order = np.arange(len(cells)); np.random.default_rng(SEED + 1).shuffle(order)
cell2fold = dict(zip(cells[order].tolist(), (np.arange(len(cells)) % K).tolist()))
meta["fold_spatial_block"] = cell.map(cell2fold).astype("Int64")
print(f"spatial blocks (2 deg): {len(cells)}")

X = conc.apply(pd.to_numeric, errors="coerce")
X = np.log10(X.clip(lower=1e-9))
X = X.fillna(X.median())
X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(X.values)
meta["fold_environmental"] = km.labels_
print(f"environmental clusters: {K}   inertia {km.inertia_:,.0f}")
print(pd.crosstab(meta.fold_environmental, meta.survey).to_string())

def haversine_min(A, B):
    R = 6371.0
    out = np.empty(len(A))
    la2, lo2 = np.radians(B[:, 0])[None, :], np.radians(B[:, 1])[None, :]
    for i in range(0, len(A), 500):
        a = A[i:i + 500]
        la1, lo1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
        h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        out[i:i + 500] = (2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).min(1)
    return out

def jaccard_min(A, B, chunk=400):
    out = np.empty(len(A)); Bs = B.sum(1)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        inter = a.astype(np.int32) @ B.T.astype(np.int32)
        union = a.sum(1)[:, None] + Bs[None, :] - inter
        out[i:i + chunk] = np.where(union > 0, 1 - inter / np.maximum(union, 1), 0.0).min(1)
    return out

M = mask.fillna(0).to_numpy(dtype=bool)
recs = []
for s in ["GEMAS", "NGSA", "NASGL"]:
    te = meta.survey.eq(s).to_numpy(); tr = ~te
    ok = (lat.notna() & lon.notna()).to_numpy()
    A = meta.loc[te & ok, ["latitude", "longitude"]].to_numpy(float)
    B = meta.loc[tr & ok, ["latitude", "longitude"]].to_numpy(float)
    for space, v, unit in (("geographic", haversine_min(A, B) if len(A) and len(B) else np.array([]), "km"),
                           ("censoring_mask_jaccard", jaccard_min(M[te], M[tr]), "unitless")):
        if not len(v):
            continue
        recs.append(dict(held_out_survey=s, space=space, unit=unit, n=int(len(v)),
                         q05=float(np.quantile(v, .05)), q25=float(np.quantile(v, .25)),
                         q50=float(np.quantile(v, .50)), q75=float(np.quantile(v, .75)),
                         q95=float(np.quantile(v, .95)),
                         mean=float(v.mean()), max=float(v.max())))
G = pd.DataFrame(recs)
print("\nNNDM target nearest-neighbour distance distributions (Ghat):")
print(G.to_string(index=False))

cols = ["row_id", "survey", "sample_id", "latitude", "longitude"]
meta[cols + ["fold_random_kfold"]].to_csv(os.path.join(OUT, "folds-random-kfold.csv"), index=False)
meta[cols + ["fold_loso"]].to_csv(os.path.join(OUT, "folds-loso.csv"), index=False)
meta[cols + ["spatial_block", "fold_spatial_block"]].to_csv(os.path.join(OUT, "folds-spatial-block.csv"), index=False)
meta[cols + ["fold_environmental"]].to_csv(os.path.join(OUT, "folds-environmental.csv"), index=False)
G.to_csv(os.path.join(OUT, "nndm-target-distributions.csv"), index=False)
pd.Series(core, name="element").to_csv(os.path.join(OUT, "primary-core-elements.csv"), index=False)

print("\nfold occupancy:")
for c in ["fold_random_kfold", "fold_spatial_block", "fold_environmental"]:
    t = pd.crosstab(meta[c], meta.survey)
    print(f"  {c}: {len(t)} folds, containing all 3 surveys = {int(t.gt(0).all(axis=1).sum())}")
print("\nseed:", SEED, "| wrote 6 files to", OUT)
