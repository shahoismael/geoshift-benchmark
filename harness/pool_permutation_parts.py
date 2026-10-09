"""Pool the split permutation-control parts into one null distribution and p-value.

Splitting the control across runs is only valid if every part tested the SAME model
against the SAME held-out set. This script verifies that before pooling: the observed
balanced accuracy must be identical across parts, and the permutation seeds must all
differ. If either check fails the parts are not exchangeable and no pooled p-value is
written.

Run after all parts are done:
  python "D:\\claude_projects\\Geochemistry\\01-paper2\\07-harness\\pool_permutation_parts.py"
"""
import glob, io, os, sys
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
R = r"D:\claude_projects\Geochemistry\01-paper2\08-results"

heads = sorted(glob.glob(os.path.join(R, "control_permutation_part*.csv")))
nulls = sorted(glob.glob(os.path.join(R, "control_permutation_null_distribution_part*.csv")))
print(f"parts found: {len(heads)} summary, {len(nulls)} null files")
if not heads or len(heads) != len(nulls):
    print("STOP: part files missing or unpaired. Nothing pooled.")
    sys.exit(2)

H = pd.concat([pd.read_csv(f) for f in heads], ignore_index=True)
print("\nper part:")
print(H[["perm_part", "perm_seed", "n_perm", "observed_balanced_accuracy",
         "permutation_mean", "p_value"]].to_string(index=False))

obs = H.observed_balanced_accuracy.round(12).unique()
if len(obs) != 1:
    print(f"\nSTOP: observed accuracy differs across parts ({obs}). The parts did not "
          "share a train/test split, so they are not exchangeable and cannot be "
          "pooled. Nothing written.")
    sys.exit(3)
if H.perm_seed.nunique() != len(H):
    print("\nSTOP: permutation seeds are not all distinct, so some parts drew the same "
          "label arrangements. Pooling them would count the same draws twice. Nothing "
          "written.")
    sys.exit(3)

observed = float(obs[0])
d = pd.concat([pd.read_csv(f) for f in nulls], ignore_index=True)
v = pd.to_numeric(d.permutation_balanced_accuracy, errors="coerce")
v = v[np.isfinite(v)].to_numpy()
n = v.size
p = float((1 + (v >= observed).sum()) / (1 + n))

print(f"\npooled permutations : {n}")
print(f"observed            : {observed:.6f}")
print(f"permutation mean    : {v.mean():.6f}")
print(f"permutation q95     : {np.quantile(v, 0.95):.6f}")
print(f"permutation max     : {v.max():.6f}")
print(f"p-value             : {p:.6f}")
verdict = ("survives stratified permutation - the signal is instrumental, not "
           "geographic" if p < 0.05 else
           "does not survive - geographic, or the test is underpowered; these are "
           "different conclusions and the power figures decide which")
print(f"verdict             : {verdict}")
if v.max() >= observed:
    print("note: at least one permutation matched or beat the observed accuracy.")

row = H.iloc[0].to_dict()
for k in ("perm_part", "perm_seed", "p_value_note", "p_value", "n_perm",
          "n_perm_finite", "permutation_mean", "permutation_q95"):
    row.pop(k, None)
row.update(n_perm_pooled=n, n_parts=len(H),
           perm_seeds=";".join(str(int(s)) for s in H.perm_seed),
           observed_balanced_accuracy=observed,
           permutation_mean=float(v.mean()),
           permutation_q95=float(np.quantile(v, 0.95)),
           permutation_max=float(v.max()),
           p_value=p, verdict=verdict)
out = os.path.join(R, "control_permutation_pooled.csv")
pd.DataFrame([row]).to_csv(out, index=False, encoding="utf-8")
pd.DataFrame({"permutation_balanced_accuracy": v}).to_csv(
    os.path.join(R, "control_permutation_null_distribution_pooled.csv"), index=False)
print(f"\nwrote {out}")
