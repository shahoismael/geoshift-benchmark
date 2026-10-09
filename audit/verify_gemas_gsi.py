import io, sys, os, glob
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
try:
    from dbfread import DBF
    HAVE = True
except Exception:
    HAVE = False
    print("dbfread missing -> pip install dbfread")

D = r"D:\claude_projects\Geochemistry\04-data\GEMAS_GSI"
OLD = r"D:\claude_projects\Geochemistry\00-paper-1\02-datasets\GEMAS"
PAIRS = [("IE_GSI_GEMAS_Agricultural_Soil_Ap_Samples_AR_XRF_EU_WGS84.dbf", "GEMAS_Ap_AquaRegia_XRF.csv"),
         ("IE_GSI_GEMAS_Agricultural_Soil_Ap_Samples_MMI_EU_WGS84.dbf",    "GEMAS_Ap_MMI.csv"),
         ("IE_GSI_GEMAS_Grazing_Land_Gr_Samples_AR_XRF_EU_WGS84.dbf",      "GEMAS_Gr_AquaRegia_XRF.csv")]

for dbf, csv in PAIRS:
    print("=" * 70)
    p = os.path.join(D, dbf)
    if HAVE:
        t = DBF(p, load=True, char_decode_errors="replace")
        new = pd.DataFrame(iter(t))
    else:
        new = None
    o = os.path.join(OLD, csv)
    old = pd.read_csv(o, low_memory=False) if os.path.exists(o) else None
    print(dbf)
    if new is not None:
        print(f"  GSI shapefile : {new.shape[0]:,} rows x {new.shape[1]} cols")
    if old is not None:
        print(f"  Paper 1 CSV   : {old.shape[0]:,} rows x {old.shape[1]} cols")
    if new is not None and old is not None:
        sn, so = set(map(str.upper, map(str, new.columns))), set(map(str.upper, map(str, old.columns)))
        print(f"  columns shared: {len(sn & so)}   only-GSI: {len(sn - so)}   only-CSV: {len(so - sn)}")
        miss = sorted(so - sn)
        if miss:
            print(f"  missing from GSI: {miss[:20]}{' ...' if len(miss) > 20 else ''}")
        # element-like columns present?
        els = [c for c in new.columns if str(c).split("_")[0] in
               ("Ag","Al","As","Ba","Be","Bi","Ca","Cd","Ce","Co","Cr","Cs","Cu","Fe","Pb","Zn")]
        print(f"  element-like columns in GSI: {len(els)}  e.g. {els[:10]}")
