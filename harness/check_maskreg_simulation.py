"""Characterise the fold-internal mask regression on simulated data (gate before real runs).

Three synthetic surveys share ONE geochemical distribution, so the only thing that
distinguishes them is survey-specific detection-limit censoring. A classifier that
recovers survey identity from the predictors is therefore reading the censoring
channel and nothing else. The check reports survey balanced accuracy from:

  true      uncensored values (reference: no channel, should sit near chance 1/3)
  half_lld  half-limit substitution         (value + pattern open)
  median    training-median imputation      (value removed, pattern open)
  maskreg   median + fold-internal mask regression (the specified treatment)
  common    every survey re-censored at the highest survey limit per element

PASS if maskreg falls to within 0.05 of the `true` reference. On FAIL the real runs
must not be launched: the treatment does not close the channel for tree models, and
that is reported as the finding.

Writes 08-results/maskreg-simulation-check.csv. Seed 20260925. Runtime ~1-2 min.
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score
from geoshift.model import CensoredLogTransform
from geoshift.maskreg import residualize_on_mask

SEED = 20260925
from geoshift.config import RESULTS
OUT = os.path.join(RESULTS, "maskreg-simulation-check.csv")
rng = np.random.default_rng(SEED)
n_per, p, S = 800, 20, 3

y = np.repeat(np.arange(S), n_per)
mu = rng.normal(1.5, 0.8, p)
logx = rng.normal(mu, 0.5, (S * n_per, p))          # identical across surveys
X = 10 ** logx
rate = rng.uniform(0.0, 0.15, (S, p))               # survey-specific censoring
lim = np.vstack([np.quantile(X, rate[s], axis=0) for s in range(S)])
L = lim[y]
mask = (X < L).astype(float)
Xc = np.where(mask == 1, np.nan, X)
D = np.where(mask == 1, L, np.nan)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
tr, te = next(sss.split(X, y))

def score(F):
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(F[tr], y[tr])
    return balanced_accuracy_score(y[te], rf.predict(F[te]))

def transformed(strategy):
    t = CensoredLogTransform(strategy=strategy, lod=D).fit(Xc[tr])
    return t.transform(Xc)

res = {"true": score(np.log10(X)),
       "half_lld": score(transformed("half_lld")),
       "median": score(transformed("median_train"))}
Xm = transformed("median_train")
res["maskreg"] = score(residualize_on_mask(Xm, mask, tr))
Lc = lim.max(axis=0)                                  # common limit per element
mc = X < Lc
t = CensoredLogTransform(strategy="median_train",
                         lod=np.where(mc, Lc, np.nan)).fit(np.where(mc, np.nan, X)[tr])
res["common"] = score(t.transform(np.where(mc, np.nan, X)))

ok = res["maskreg"] <= res["true"] + 0.05
verdict = "PASS" if ok else "FAIL"
print(f"common-limit closes the channel: {res['common'] <= res['true'] + 0.05}")
print(f"censored cells: {100*mask.mean():.2f}%   chance: 0.333")
for k, v in res.items():
    print(f"  {k:9s} survey balanced accuracy = {v:.3f}")
print(f"VERDICT: {verdict} (maskreg must be <= true + 0.05 = {res['true']+0.05:.3f})")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["arm", "survey_balanced_accuracy"])
    for k, v in res.items(): w.writerow([k, f"{v:.4f}"])
    w.writerow(["verdict", verdict])
sys.exit(0 if ok else 3)
