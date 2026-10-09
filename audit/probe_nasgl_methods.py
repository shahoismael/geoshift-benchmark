import io, sys, os, re
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
NAS = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\NASGL"
INT = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
SEVEN = ["Fe", "Hg", "In", "Li", "Se", "Te", "Tl"]

f = os.path.join(NAS, "Appendix_3b_Ahorizon_18Sept2013.txt")
with open(f, encoding="latin-1", errors="replace") as fh:
    head = "".join(fh.readline() for _ in range(20))

print("############ digestion / method language in the NASGL header block")
for kw in ["acid", "digest", "aqua", "regia", "ICP", "AES", "MS", "XRF", "cold vapor",
           "cold-vapour", "combustion", "hydride", "HF", "HCl", "HNO3", "HClO4", "total"]:
    for m in re.finditer(kw, head, re.I):
        s = max(0, m.start() - 130); e = min(len(head), m.end() + 130)
        snippet = " ".join(head[s:e].split())
        print(f"  [{kw}] ...{snippet}...")
        break
print()

print("############ which of the seven are in NASGL's interim table")
n = pd.read_parquet(os.path.join(INT, "nasgl_conc.parquet"))
g = pd.read_parquet(os.path.join(INT, "gemas_conc.parquet"))
ng = pd.read_parquet(os.path.join(INT, "ngsa_conc.parquet"))
print("NASGL elements:", sorted(n.columns))
print()
for t in SEVEN:
    print(f"  {t:3s}  GEMAS {t in g.columns!s:5s}  NGSA {t in ng.columns!s:5s}  NASGL {t in n.columns!s:5s}")
print()
inall = [t for t in SEVEN if t in n.columns and t in g.columns and t in ng.columns]
print("in all three:", inall)
print("NOT in NASGL:", [t for t in SEVEN if t not in n.columns])
print()

print("############ median concentration comparison for the shared ones")
for t in inall:
    a = pd.to_numeric(g[t], errors="coerce"); b = pd.to_numeric(ng[t], errors="coerce")
    c = pd.to_numeric(n[t], errors="coerce")
    print(f"  {t:3s} median  GEMAS {np.nanmedian(a):>10.4g}   NGSA {np.nanmedian(b):>10.4g}"
          f"   NASGL {np.nanmedian(c):>10.4g}    NASGL/GEMAS {np.nanmedian(c)/np.nanmedian(a):>7.3f}")
