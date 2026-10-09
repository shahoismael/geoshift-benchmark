import io, sys, os, re, glob, zipfile
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NAS = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\NASGL"
GEM = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\GEMAS"
NGS = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\NGSA"
INT = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\interim"
SEVEN = ["Fe", "Hg", "In", "Li", "Se", "Te", "Tl"]

print("############ NASGL A-horizon: find the real header row")
f = os.path.join(NAS, "Appendix_3b_Ahorizon_18Sept2013.txt")
with open(f, encoding="latin-1", errors="replace") as fh:
    lines = [fh.readline() for _ in range(8)]
for i, L in enumerate(lines):
    print(f"  row {i}: {L[:220].rstrip()}")
print()

for hdr in (1, 2, 3):
    try:
        d = pd.read_csv(f, sep="\t", skiprows=hdr, nrows=5, low_memory=False, encoding="latin-1")
        cols = [str(c) for c in d.columns]
        if sum(c.upper().startswith(("FE", "HG", "LI", "SE", "TE", "TL", "IN")) for c in cols) >= 3:
            print(f"  -> header at skiprows={hdr}; {len(cols)} columns")
            for t in SEVEN:
                print(f"     {t:3s} -> {[c for c in cols if c.upper().startswith(t.upper())]}")
            print("     all columns:", cols)
            break
    except Exception as e:
        print(f"  skiprows={hdr} failed: {type(e).__name__}")
print()

print("############ Which GEMAS Fe column did Paper 1 use?")
gi = pd.read_parquet(os.path.join(INT, "gemas_conc.parquet"))
g  = pd.read_csv(os.path.join(GEM, "GEMAS_Ap_AquaRegia_XRF.csv"), low_memory=False)
if "Fe" in gi.columns:
    a = pd.to_numeric(gi["Fe"], errors="coerce")
    for cand in ["FE", "FE_XRF", "FE2O3"]:
        if cand in g.columns:
            b = pd.to_numeric(g[cand], errors="coerce")
            n = min(len(a), len(b))
            r = np.corrcoef(a[:n].fillna(0), b[:n].fillna(0))[0, 1]
            print(f"  interim Fe vs GEMAS {cand:7s}: median ratio "
                  f"{np.nanmedian(a[:n]/b[:n]):.4g}   corr {r:.4f}")
print()

print("############ CRM search across all documentation")
FAMS = r"(OREAS|STSD|GXR|SRM\s?\d|NIST|GBW|SdAR|AMIS|CANMET|TILL-?\d|SO-?\d|BCR-?\d|" \
       r"GSP-?\d|GSD-?\d|JSd-?\d|NCS\s?DC|MESS|PACS|LKSD|SY-?\d|MRG-?\d|W-?2a|BHVO|AGV)"
targets = []
targets += glob.glob(os.path.join(NGS, "*.pdf"))
targets += glob.glob(os.path.join(NAS, "*.txt"))
targets += glob.glob(os.path.join(GEM, "*.csv"))
targets += glob.glob(r"D:\claude_projects\Geochemistry\04-data\GEMAS_GSI\*.xml")
targets += glob.glob(r"D:\claude_projects\Geochemistry\04-data\TELLUS\topsoil_A\*.txt")

try:
    from pypdf import PdfReader
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False
    print("  (pypdf missing - PDFs skipped)")

for t in targets:
    hits = {}
    try:
        if t.lower().endswith(".pdf"):
            if not HAVE_PDF:
                continue
            rd = PdfReader(t)
            txt = "\n".join((p.extract_text() or "") for p in rd.pages[:80])
        else:
            with open(t, encoding="latin-1", errors="replace") as fh:
                txt = fh.read(6_000_000)
        for m in re.finditer(FAMS, txt, re.I):
            hits[m.group(0).upper()] = hits.get(m.group(0).upper(), 0) + 1
    except Exception as e:
        print(f"  {os.path.basename(t)}: read failed {type(e).__name__}")
        continue
    if hits:
        top = sorted(hits.items(), key=lambda kv: -kv[1])[:12]
        print(f"  {os.path.basename(t)}: {top}")
    else:
        print(f"  {os.path.basename(t)}: no CRM identifiers found")
