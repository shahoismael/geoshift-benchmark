"""
Tellus Shallow Topsoil A - canonical censoring-mask construction.

Supersedes check_censoring.py and paired_method_contrast.py.

Corrections applied:
  C1  Units_Detection_Limits contains duplicate analyte rows (Accredited Yes/No)
      with LLDs differing by up to 1e4. Select the accredited row explicitly.
  C2  Use the PUBLISHED HDL column (the substituted half-limit) rather than a
      recomputed 0.5*LLD, which is not representable at the printed precision
      of 44 of 111 columns.
  C3  Flag left-censoring as (value == HDL) OR (value < LLD). The survey does
      not substitute uniformly; thousands of sub-LLD values are not equal to HDL.
  C4  Flag right-censoring at the published URL (upper reporting limit); the
      sheet states ">ULR values replaced with URL values".
"""
import io, sys, os, re, json
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

XL  = r"D:\claude_projects\Geochemistry\04-data\TELLUS\topsoil_A\58xxxxA-65xxxxA_Shallow_Topsoil_Download_v1.2.xlsx"
OUT = r"D:\claude_projects\Geochemistry\01-paper2\03-crosswalk"
os.makedirs(OUT, exist_ok=True)

d = pd.read_excel(XL, sheet_name="58xxxxS-65xxxxA")
u = pd.read_excel(XL, sheet_name="Units_Detection_Limits")
u.columns = [str(c).strip() for c in u.columns]
u["Analyte"] = u["Analyte"].astype(str).str.strip()
u["Accredited"] = u["Accredited"].astype(str).str.strip().str.lower()
for c in ("LLD", "HDL", "URL"):
    u[c] = pd.to_numeric(u[c], errors="coerce")
u = u[u["Analyte"].str.contains("_", na=False)]

# ---- C1: one limit row per analyte, accreditation preferred, provenance kept
limits = {}
for a, g in u.groupby("Analyte", sort=False):
    acc = g[g["Accredited"] == "yes"]
    row, prov = (acc.iloc[0], "accredited") if len(acc) else (g.iloc[0], "non_accredited")
    limits[a] = dict(LLD=row["LLD"], HDL=row["HDL"], URL=row["URL"],
                     limit_provenance=prov, n_limit_rows=len(g),
                     lld_spread=(g["LLD"].max() / g["LLD"].min()
                                 if g["LLD"].min() and np.isfinite(g["LLD"].min()) else np.nan))

# ---- element symbol per column, oxide forms resolved explicitly
OXIDE = {"SiO2": "Si", "TiO2": "Ti", "Al2O3": "Al", "Fe2O3": "Fe", "MnO": "Mn",
         "MgO": "Mg", "CaO": "Ca", "Na2O": "Na", "K2O": "K", "P2O5": "P", "SO3": "S"}
def symbol(col):
    stem = col.split("_")[0].strip()
    if stem in OXIDE:
        return OXIDE[stem], True
    return (stem, False) if re.fullmatch(r"[A-Z][a-z]?", stem) else (None, False)

# ---- C2/C3/C4: build the mask
recs, masks = [], {}
for col in d.columns:
    c = str(col).strip()
    if c not in limits:
        continue
    L, H, U = limits[c]["LLD"], limits[c]["HDL"], limits[c]["URL"]
    v = pd.to_numeric(d[col], errors="coerce")
    obs = v.notna()
    n = int(obs.sum())
    if n == 0 or not np.isfinite(L):
        continue
    if not np.isfinite(H):
        H = 0.5 * L
    at_hdl = np.isclose(v, H, rtol=1e-6, atol=1e-12) & obs
    sub    = (v < L * (1 - 1e-9)) & obs
    left   = (at_hdl | sub)
    right  = (np.isclose(v, U, rtol=1e-6, atol=1e-12) & obs) if np.isfinite(U) else pd.Series(False, index=v.index)
    sym, is_ox = symbol(c)
    route = "XRFS" if c.endswith("XRFS") else ("ICPar" if c.endswith("ICPar") else "other")
    masks[c] = dict(left=left.fillna(False), right=right.fillna(False), obs=obs)
    recs.append(dict(column=c, element=sym, oxide_form=is_ox, route=route, n=n,
                     LLD=L, HDL=H, URL=U, **{k: limits[c][k] for k in
                     ("limit_provenance", "n_limit_rows", "lld_spread")},
                     n_left=int(left.sum()), pct_left=round(100*int(left.sum())/n, 3),
                     n_at_HDL=int(at_hdl.sum()), n_below_LLD_not_HDL=int((sub & ~at_hdl).sum()),
                     n_right=int(right.sum()), pct_right=round(100*int(right.sum())/n, 3)))

cat = pd.DataFrame(recs)
cat.to_csv(os.path.join(OUT, "tellus-censoring-catalogue.csv"), index=False)

# ---- paired contrast on the corrected mask
pairs = []
for sym, g in cat[cat.route.isin(["XRFS", "ICPar"])].dropna(subset=["element"]).groupby("element"):
    x = g[g.route == "XRFS"]; i = g[g.route == "ICPar"]
    if len(x) != 1 or len(i) != 1:
        continue
    xc, ic = x.iloc[0]["column"], i.iloc[0]["column"]
    ok = masks[xc]["obs"] & masks[ic]["obs"]
    n = int(ok.sum())
    if n == 0:
        continue
    fx, fi = masks[xc]["left"][ok], masks[ic]["left"][ok]
    pairs.append(dict(element=sym, n=n, xrfs_col=xc, icpar_col=ic,
                      xrfs_oxide=bool(x.iloc[0]["oxide_form"]),
                      xrfs_limit=x.iloc[0]["limit_provenance"],
                      icpar_limit=i.iloc[0]["limit_provenance"],
                      xrfs_pct=round(100*int(fx.sum())/n, 2),
                      icpar_pct=round(100*int(fi.sum())/n, 2),
                      disagree_pct=round(100*int((fx ^ fi).sum())/n, 2),
                      xrfs_right=int(masks[xc]["right"][ok].sum()),
                      icpar_right=int(masks[ic]["right"][ok].sum())))
p = pd.DataFrame(pairs).sort_values("disagree_pct", ascending=False)
p.to_csv(os.path.join(OUT, "tellus-paired-method-contrast.csv"), index=False)

# ---- M7: best-single-element route null
best = p.assign(sep=(p.xrfs_pct - p.icpar_pct).abs()).sort_values("sep", ascending=False)
acc1 = []
for _, r in best.head(5).iterrows():
    ok = masks[r.xrfs_col]["obs"] & masks[r.icpar_col]["obs"]
    fx, fi = masks[r.xrfs_col]["left"][ok], masks[r.icpar_col]["left"][ok]
    # rule: predict XRFS when censored, if XRFS censors more; balanced accuracy
    sgn = 1 if r.xrfs_pct >= r.icpar_pct else -1
    tpr = (fx if sgn == 1 else ~fx).mean(); tnr = (~fi if sgn == 1 else fi).mean()
    acc1.append(dict(element=r.element, rule_balanced_accuracy=round(float((tpr+tnr)/2), 4),
                     xrfs_pct=r.xrfs_pct, icpar_pct=r.icpar_pct))
pd.DataFrame(acc1).to_csv(os.path.join(OUT, "tellus-single-element-route-null.csv"), index=False)

# ---- report
prov = cat.limit_provenance.value_counts().to_dict()
dupe = cat[cat.n_limit_rows > 1]
print("columns with limits            :", len(cat))
print("limit provenance               :", prov)
print("columns with duplicate limit rows:", len(dupe),
      "| max LLD spread:", f"{dupe.lld_spread.max():,.0f}x" if len(dupe) else "-")
print()
print("LEFT-censoring recoverability")
print("  columns with >0 left-censored :", int((cat.n_left > 0).sum()), "of", len(cat))
print("  cells at published HDL        :", int(cat.n_at_HDL.sum()))
print("  cells below LLD but NOT at HDL:", int(cat.n_below_LLD_not_HDL.sum()),
      "  <- missed by the value==0.5*LLD rule")
print()
print("RIGHT-censoring (URL substitution)")
print("  columns with an upper limit   :", int(cat.URL.notna().sum()))
print("  columns with >0 right-censored:", int((cat.n_right > 0).sum()))
print("  right-censored cells          :", int(cat.n_right.sum()))
print()
print("PAIRED CONTRAST (corrected)")
print("  elements paired               :", len(p))
print("  oxide-form XRFS pairings      :", int(p.xrfs_oxide.sum()))
print("  both limits accredited        :", int(((p.xrfs_limit=='accredited')&(p.icpar_limit=='accredited')).sum()))
print("  disagree on >5% of samples    :", int((p.disagree_pct > 5).sum()), "of", len(p))
print("  median disagreement           :", f"{p.disagree_pct.median():.2f}%")
print("  mean XRFS / ICPar censoring   :", f"{p.xrfs_pct.mean():.2f}% / {p.icpar_pct.mean():.2f}%")
print("  elements where ICPar censors more:", int((p.icpar_pct > p.xrfs_pct).sum()))
print()
print("  top 10 disagreement:")
print(p.head(10)[["element","n","xrfs_pct","icpar_pct","disagree_pct","xrfs_limit","icpar_limit"]].to_string(index=False))
print()
print("  M7 best-single-element route null:")
print(pd.DataFrame(acc1).to_string(index=False))
print("\nwrote 3 files to", OUT)
