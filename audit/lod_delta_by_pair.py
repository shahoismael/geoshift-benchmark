import io, sys, os
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUT = r"D:\claude_projects\Geochemistry\01-paper2\03-crosswalk"
d = pd.read_csv(os.path.join(OUT, "lod-disparity-delta.csv"), index_col=0)

# the 39-element primary core = 44 shared, less the five set aside
FIVE = ["Li", "In", "Te", "Tl", "Se"]
cw = pd.read_csv(os.path.join(OUT, "element-crosswalk.csv"))
col = next((c for c in cw.columns if c.lower() in ("element", "el", "symbol")), cw.columns[0])
shared = [e for e in cw[col].astype(str) if e in d.index]
core = [e for e in shared if e not in FIVE]
print(f"crosswalk elements found in the limit table: {len(shared)}   primary core: {len(core)}\n")

PAIRS = [("NGSA_over_NASGL", "NGSA <-> NASGL"),
         ("GEMAS_over_NGSA", "GEMAS <-> NGSA"),
         ("GEMAS_over_NASGL", "GEMAS <-> NASGL")]

for col_, name in PAIRS:
    r = d.loc[[e for e in core if e in d.index], col_].dropna()
    f = np.maximum(r, 1 / r)
    print(f"===== {name}   (primary-core elements with Delta defined: {len(f)})")
    if len(f) == 0:
        print("   Delta UNDEFINED for every core element on this fold -> Task 3 cannot run here\n")
        continue
    for th in (2, 5, 10, 20, 100):
        print(f"   Delta >= {th:>3}x : {int((f >= th).sum()):>3} of {len(f)}"
              f"   ({100*int((f>=th).sum())/len(f):.0f}%)")
    print(f"   median {f.median():.2f}x   IQR {f.quantile(.25):.2f}-{f.quantile(.75):.2f}x"
          f"   max {f.max():.0f}x ({f.idxmax()})")
    print(f"   elements: {sorted(f.index.tolist())}\n")

print("===== verdict for Task 3")
ngsa_nasgl = d.loc[[e for e in core if e in d.index], "NGSA_over_NASGL"].dropna()
gem_any = d.loc[[e for e in core if e in d.index],
                ["GEMAS_over_NGSA", "GEMAS_over_NASGL"]].notna().any(axis=1).sum()
print(f"  NGSA<->NASGL fold : Delta defined for {len(ngsa_nasgl)} core elements -> Task 3 RUNS here")
print(f"  any GEMAS fold    : Delta defined for {gem_any} core elements -> Task 3 cannot be the"
      f" primary design on GEMAS folds")
