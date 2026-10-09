"""Feasibility of common-limit re-censoring on the real observational arm.

Common-limit re-censoring sets, per element, one limit for all three surveys (the
highest survey limit) and censors every value below it. Censoring then follows one
rule everywhere and carries geochemistry, not laboratory. In simulation it closes
the channel (survey accuracy at chance) where the specified mask regression does not.

This script measures what it would cost on the real data: per core element, whether
every survey's limit is known, and the censoring rate per survey before and after.
Writes 08-results/common-limit-feasibility.csv. Runtime under a minute.
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geoshift.data import load_observational, core_elements

from geoshift.config import RESULTS
OUT = os.path.join(RESULTS, "common-limit-feasibility.csv")
els = core_elements()
meta, conc, mask, lod = load_observational(els)
sv = meta["survey"].to_numpy()
S = ["GEMAS", "NGSA", "NASGL"]
rows = []
for e in els:
    lim = {}
    for s in S:
        sel = (sv == s) & (mask[e].to_numpy() == 1)
        v = lod[e].to_numpy(float)[sel]
        v = v[np.isfinite(v) & (v > 0)]
        lim[s] = float(np.median(v)) if v.size else np.nan
    known = [s for s in S if np.isfinite(lim[s])]
    Lc = max(lim[s] for s in known) if known else np.nan
    r = {"element": e, "limits_known": len(known), "common_limit": Lc}
    x = conc[e].to_numpy(float); m = mask[e].to_numpy() == 1
    for s in S:
        sel = sv == s
        before = m[sel].mean()
        after = (m[sel] | (np.isfinite(x[sel]) & (x[sel] < Lc))).mean() if np.isfinite(Lc) else before
        r[f"{s}_before"] = round(100 * before, 2); r[f"{s}_after"] = round(100 * after, 2)
    r["max_after"] = max(r[f"{s}_after"] for s in S)
    rows.append(r)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
all3 = [r for r in rows if r["limits_known"] == 3]
ok25 = [r for r in all3 if r["max_after"] <= 25]
ok50 = [r for r in all3 if r["max_after"] <= 50]
print(f"core elements: {len(rows)}")
print(f"limit known in all three surveys: {len(all3)}")
print(f"  of those, re-censoring stays <=25% in every survey: {len(ok25)}  {[r['element'] for r in ok25]}")
print(f"  of those, re-censoring stays <=50% in every survey: {len(ok50)}")
print(f"wrote {OUT}")
