"""Build the GeoShift element crosswalk from Paper 1 interim tables.

Outputs, into 01-paper2/03-crosswalk/:
  element-crosswalk.csv   one row per element seen in any survey
  element-exclusions.csv  elements not carried into the 44-element core
  crosswalk-summary.txt   counts and the NGSA method selection actually used
"""
import pandas as pd, os, numpy as np

SRC = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets"
INT = os.path.join(SRC, "interim")
PRO = os.path.join(SRC, "processed")
OUT = r"D:\claude_projects\Geochemistry\01-paper2\03-crosswalk"

gemas = pd.read_parquet(os.path.join(INT, "gemas_conc.parquet"))
ngsa = pd.read_parquet(os.path.join(INT, "ngsa_conc.parquet"))
nasgl = pd.read_parquet(os.path.join(INT, "nasgl_conc.parquet"))
pooled = pd.read_parquet(os.path.join(PRO, "pooled_conc.parquet"))
cat = pd.read_parquet(os.path.join(INT, "ngsa_method_catalogue.parquet"))

g, n, a = set(gemas.columns), set(ngsa.columns), set(nasgl.columns)
core = list(pooled.columns)
allel = sorted(g | n | a)

# NGSA: which method was actually selected per element (rank 1 = chosen)
chosen = cat.sort_values(["element", "rank"]).groupby("element").first()

# censoring rate per element per survey, from the mask tables
masks = {}
for s, f in [("gemas", "gemas_mask"), ("ngsa", "ngsa_mask"), ("nasgl", "nasgl_mask")]:
    masks[s] = pd.read_parquet(os.path.join(INT, f + ".parquet"))

# LOD: modal reported detection limit per element per survey
lods = {}
for s, f in [("gemas", "gemas_lod"), ("ngsa", "ngsa_lod"), ("nasgl", "nasgl_lod")]:
    lods[s] = pd.read_parquet(os.path.join(INT, f + ".parquet"))

rows = []
for el in allel:
    r = {"element": el,
         "in_gemas": el in g, "in_ngsa": el in n, "in_nasgl": el in a,
         "in_core_44": el in core}
    r["n_surveys"] = int(r["in_gemas"]) + int(r["in_ngsa"]) + int(r["in_nasgl"])
    # NGSA selected method
    if el in chosen.index:
        r["ngsa_method"] = chosen.loc[el, "method"]
        r["ngsa_unit"] = chosen.loc[el, "unit"]
        r["ngsa_lod_reported"] = chosen.loc[el, "lod_reported"]
        r["ngsa_source_column"] = chosen.loc[el, "source_column"]
    else:
        r["ngsa_method"] = r["ngsa_unit"] = r["ngsa_lod_reported"] = r["ngsa_source_column"] = None
    # censoring rate and modal LOD
    for s in ["gemas", "ngsa", "nasgl"]:
        m, l = masks[s], lods[s]
        if el in m.columns:
            r[f"cens_rate_{s}"] = round(float(np.nanmean(m[el].astype(float))), 4)
        else:
            r[f"cens_rate_{s}"] = None
        if el in l.columns:
            v = pd.to_numeric(l[el], errors="coerce").dropna()
            r[f"lod_{s}"] = float(v.mode().iloc[0]) if len(v) else None
        else:
            r[f"lod_{s}"] = None
    # LOD disparity across the surveys that have the element
    vals = [r[f"lod_{s}"] for s in ["gemas", "ngsa", "nasgl"] if r[f"lod_{s}"]]
    r["lod_ratio_max"] = round(max(vals) / min(vals), 3) if len(vals) > 1 else None
    r["digestion_equivalent"] = ""   # Dashti signs off
    r["reviewer_note"] = ""
    rows.append(r)

cw = pd.DataFrame(rows)
cw = cw.sort_values(["in_core_44", "n_surveys", "element"],
                    ascending=[False, False, True])
cw.to_csv(os.path.join(OUT, "element-crosswalk.csv"), index=False)

exc = cw[~cw["in_core_44"]].copy()
exc["exclusion_reason"] = np.where(
    exc["n_surveys"] < 3,
    "not reported in all three surveys",
    "present in all three but dropped from the Paper 1 core — reason to confirm")
exc[["element", "in_gemas", "in_ngsa", "in_nasgl",
     "n_surveys", "exclusion_reason"]].to_csv(
    os.path.join(OUT, "element-exclusions.csv"), index=False)

lines = []
lines.append(f"elements seen in any survey : {len(allel)}")
lines.append(f"GEMAS {len(g)} | NGSA {len(n)} | NASGL {len(a)}")
lines.append(f"present in all three        : {sum(cw.n_surveys == 3)}")
lines.append(f"Paper 1 core (pooled_conc)  : {len(core)}")
lines.append(f"excluded                    : {len(exc)}")
lines.append("")
lines.append("NGSA method selected per core element (rank 1 of the catalogue):")
sub = cw[cw.in_core_44][["element", "ngsa_method", "ngsa_unit"]]
for _, x in sub.iterrows():
    lines.append(f"  {x.element:<3} {str(x.ngsa_method):<10} {x.ngsa_unit}")
lines.append("")
lines.append("NGSA method mix across the core:")
lines.append(str(sub.ngsa_method.value_counts()))
lines.append("")
d = cw[cw.in_core_44 & cw.lod_ratio_max.notna()]
lines.append(f"core elements with LOD ratio >= 2x : {int((d.lod_ratio_max >= 2).sum())} of {len(d)}")
txt = "\n".join(lines)
open(os.path.join(OUT, "crosswalk-summary.txt"), "w", encoding="utf-8").write(txt)
print(txt)
