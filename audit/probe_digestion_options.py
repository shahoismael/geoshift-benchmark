"""Decision 1 evidence: for the 7 NGSA aqua-regia elements, what ELSE is available
in each survey? Aligning digestion class per element beats keeping-with-caveat."""
import io, sys, os
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
pd.set_option("display.width", 200)

INT  = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
GEM  = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\GEMAS"
NAS  = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\NASGL"
SEVEN = ["Fe", "Hg", "In", "Li", "Se", "Te", "Tl"]

print("################ 1. NGSA method catalogue — ALL methods per element")
cat = pd.read_parquet(os.path.join(INT, "ngsa_method_catalogue.parquet"))
print("catalogue cols:", list(cat.columns), "| rows:", len(cat))
sub = cat[cat["element"].isin(SEVEN)].sort_values(["element", "rank"])
print(sub.to_string(index=False))
print()
print("--- for comparison, two elements Paper 1 assigned to ICP-MS total ---")
print(cat[cat["element"].isin(["Cu", "Pb"])].sort_values(["element", "rank"]).to_string(index=False))
print()

print("################ 2. NGSA all-methods concentration table")
am = pd.read_parquet(os.path.join(INT, "ngsa_allmethods_conc.parquet"))
print("shape:", am.shape)
cols7 = [c for c in am.columns if str(c).split("_")[0] in SEVEN]
print("columns for the seven:", cols7)
if cols7:
    print(am[cols7].notna().sum().to_string())
print()

print("################ 3. GEMAS Ap AquaRegia+XRF — which columns, which digestion")
g = pd.read_csv(os.path.join(GEM, "GEMAS_Ap_AquaRegia_XRF.csv"), nrows=3, low_memory=False)
gc = [str(c) for c in g.columns]
print("total cols:", len(gc))
for tag in ["Fe", "Hg", "In", "Li", "Se", "Te", "Tl"]:
    hits = [c for c in gc if c.upper().startswith(tag.upper())]
    print(f"  {tag:3s} -> {hits}")
print()
print("  all columns containing a digestion hint:")
hint = [c for c in gc if any(k in c.upper() for k in ("AR", "XRF", "TOT", "MMI", "_AQ"))]
print("  ", hint[:60])
print()

print("################ 4. NASGL A-horizon columns")
import glob
f = os.path.join(NAS, "Appendix_3b_Ahorizon_18Sept2013.txt")
head = pd.read_csv(f, sep="\t", nrows=3, low_memory=False, encoding="latin-1")
nc = [str(c) for c in head.columns]
print("total cols:", len(nc))
for tag in SEVEN:
    hits = [c for c in nc if c.upper().startswith(tag.upper())]
    print(f"  {tag:3s} -> {hits}")
print()
print("  first 40 columns:", nc[:40])
