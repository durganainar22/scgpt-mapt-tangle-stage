#!/usr/bin/env python
"""
v3 Step 4c: is the selected model (A = mean of 8 neuron-subclass pseudobulk models, dev AUC 0.729) sound,
or overfit / lucky / confounded? Development donors only; the locked test donors are dropped on load.
Nothing here changes model A - these are diagnostics, reported as they come out.

  1 train-vs-CV gap     in-sample AUC of each subclass model vs its out-of-fold AUC
  2 regularisation      chosen C per subclass/fold (at the grid edge = warning)
  3 repeated CV         20 different stratified 5-fold splits of the 67 donors -> AUC distribution
  4 calibration         Brier score vs base-rate Brier; observed rate per predicted-probability bin
  5 confounders         Spearman of P(V-VI) with age, PMI, RIN, neurons per donor, APOE4, sex, chemistry;
                        likelihood-ratio test: label ~ covariates  vs  label ~ covariates + logit(P)
  6 stability           AUC vs number of genes (500-5000); top-gene overlap across folds

Inputs:  data/seaad_pseudobulk_subclass.h5ad, seaad/results/01_donors.csv, 02_splits.csv
Output:  seaad/results/04c_checks.txt

Usage:   python seaad/04c_checks.py      (uses SLURM_CPUS_PER_TASK cores)
"""
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
OUT = 'seaad/results'
SEED, MIN_CPM, MIN_CELLS, N_REPEATS = 42, 10, 20, 20
CS = np.logspace(-4, 2, 13)
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
NEURON_SUBCLASSES = ['L2/3 IT', 'L4 IT', 'L5 IT', 'L6 IT', 'Vip', 'Pvalb', 'Sst', 'Lamp5']

splits = pd.read_csv(f'{OUT}/02_splits.csv', index_col='Donor ID')
dev = splits[splits['split'] != 'test']
y = dev['label'].astype(int)
fold0 = dev['split'].str.replace('fold', '').astype(int)
meta = pd.read_csv(f'{OUT}/01_donors.csv', index_col='Donor ID').loc[y.index]
pbn = ad.read_h5ad('data/seaad_pseudobulk_subclass.h5ad')

lines = []


def say(s=''):
    print(s, flush=True)
    lines.append(str(s))


def log_cpm(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e6)


SUB = {}
for sc in NEURON_SUBCLASSES:
    s = pbn[(pbn.obs['Subclass'].astype(str) == sc) & (pbn.obs['n_cells'] >= MIN_CELLS)]
    s = s[s.obs['Donor ID'].isin(y.index)]
    SUB[sc] = pd.DataFrame(log_cpm(s.X), index=s.obs['Donor ID'].to_numpy(), columns=pbn.var_names)


def top_genes(Xtr, n_hvg):
    expressed = np.expm1(Xtr).mean(axis=0) >= MIN_CPM
    var = np.where(expressed, Xtr.var(axis=0), -np.inf)
    return np.argsort(var)[::-1][:min(n_hvg, int(expressed.sum()))]


def fit(Xtr, ytr):
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=CS, cv=3, penalty='l2', scoring='roc_auc', class_weight='balanced',
        max_iter=5000, random_state=SEED)).fit(Xtr, ytr)


def model_a(folds, n_hvg=2000, detail=False):
    """Model A out-of-fold P for the given donor->fold map; optionally per-fold diagnostics."""
    probs, diag = {}, []
    for sc, X in SUB.items():
        p = pd.Series(np.nan, index=y.index)
        for k in sorted(folds.unique()):
            tr = [d for d in y.index[folds != k] if d in X.index]
            te = [d for d in y.index[folds == k] if d in X.index]
            g = top_genes(X.loc[tr].to_numpy(), n_hvg)
            m = fit(X.loc[tr].to_numpy()[:, g], y[tr])
            p[te] = m.predict_proba(X.loc[te].to_numpy()[:, g])[:, 1]
            if detail:
                lrm = m[-1]
                diag.append({'subclass': sc, 'fold': k, 'C': lrm.C_[0],
                             'train_auc': roc_auc_score(y[tr], m.predict_proba(X.loc[tr].to_numpy()[:, g])[:, 1]),
                             'top_genes': set(X.columns[g][np.argsort(-np.abs(lrm.coef_[0]))[:50]])})
        probs[sc] = p
    pa = pd.DataFrame(probs).mean(axis=1)
    return (pa, pd.DataFrame(probs), pd.DataFrame(diag)) if detail else pa


say('Model A diagnostics - development donors only (locked test untouched)')
pa, per_sub, diag = model_a(fold0, detail=True)
say(f'Model A out-of-fold AUC (original folds): {roc_auc_score(y, pa):.3f}')

# 1 + 2 ------------------------------------------------------------------------------------------
say('\n=== 1. Train vs out-of-fold AUC, and 2. chosen regularisation C (grid 1e-4 .. 1e2) ===')
rows = []
for sc in NEURON_SUBCLASSES:
    d = diag[diag.subclass == sc]
    ok = per_sub[sc].notna()
    rows.append({'subclass': sc, 'train_auc_mean': d.train_auc.mean(),
                 'oof_auc': roc_auc_score(y[ok], per_sub[sc][ok]),
                 'C_per_fold': ' '.join(f'{c:.0e}' for c in d.C),
                 'C_at_edge': int(((d.C == CS[0]) | (d.C == CS[-1])).sum())})
say(pd.DataFrame(rows).set_index('subclass').round(3).to_string())
say('Reading: with 2000 genes and ~53 training donors a linear model can always separate the training '
    'donors (a label-shuffled null run also gave train AUC ~1.0), so train AUC ~1.0 is expected, not by itself '
    'memorisation - the out-of-fold AUC, repeated CV (3) and permutation test are what count. '
    'C repeatedly at an edge of the grid means the grid, not the data, is setting the model.')

# 3 ----------------------------------------------------------------------------------------------
say(f'\n=== 3. Repeated CV: {N_REPEATS} different stratified 5-fold splits of the 67 donors ===')
strata = y.astype(str) + '_' + dev['dementia'].astype(str)


def one_repeat(r):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + r)
    f = pd.Series(-1, index=y.index)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), strata)):
        f.iloc[te] = k
    return roc_auc_score(y, model_a(f))


rep = np.array(Parallel(n_jobs=N_JOBS)(delayed(one_repeat)(r) for r in range(N_REPEATS)))
say(f'AUC over repeats: mean {rep.mean():.3f}, SD {rep.std(ddof=1):.3f}, '
    f'min {rep.min():.3f}, 5th-95th pct {np.percentile(rep, 5):.3f}-{np.percentile(rep, 95):.3f}')
say(f'Original folds gave {roc_auc_score(y, pa):.3f} -> '
    f'{"within" if rep.min() <= roc_auc_score(y, pa) <= rep.max() else "OUTSIDE"} the repeat range')

# 4 ----------------------------------------------------------------------------------------------
say('\n=== 4. Calibration ===')
base = np.full(len(y), y.mean())
say(f'Brier: model A {brier_score_loss(y, pa):.3f}  vs base rate {brier_score_loss(y, base):.3f} (lower is better)')
bins = pd.cut(pa, [0, 0.3, 0.45, 0.55, 0.7, 1.0])
say(pd.DataFrame({'n': y.groupby(bins).size(), 'mean_pred': pa.groupby(bins).mean(),
                  'observed_V-VI': y.groupby(bins).mean()}).round(2).to_string())

# 5 ----------------------------------------------------------------------------------------------
say('\n=== 5. Confounders ===')
cv = pd.DataFrame(index=y.index)
cv['age'] = pd.to_numeric(meta['Age at Death'], errors='coerce')
cv['PMI'] = pd.to_numeric(meta['PMI'], errors='coerce')
cv['RIN'] = pd.to_numeric(meta['RIN'], errors='coerce')
cv['neurons'] = meta['n_neurons_used']
cv['APOE4'] = meta['APOE Genotype'].astype(str).str.count('4')
rows = []
for c in cv.columns:
    ok = cv[c].notna()
    r_all = spearmanr(pa[ok], cv[c][ok])
    wi = [spearmanr(pa[ok & (y == g)], cv[c][ok & (y == g)]).correlation for g in (0, 1)]
    rows.append({'variable': c, 'rho_with_P': r_all.correlation, 'p': r_all.pvalue,
                 'rho_within_0-IV': wi[0], 'rho_within_V-VI': wi[1]})
for c, lab in [('Sex', 'Male'), ('methods', '10Xv3.1|10xMulti')]:
    g = meta[c].astype(str) == lab
    rows.append({'variable': f'{c}={lab}', 'rho_with_P': np.nan,
                 'p': mannwhitneyu(pa[g], pa[~g]).pvalue, 'rho_within_0-IV': np.nan, 'rho_within_V-VI': np.nan})
say(pd.DataFrame(rows).set_index('variable').round(3).to_string())
say('Reading: rho_within_* shows whether P tracks the variable among donors with the SAME label '
    '(a technical driver would show up here).')
for c, lab in [('methods', '10Xv3.1|10xMulti'), ('Sex', 'Male')]:
    g = meta[c].astype(str) == lab
    for name, m_ in [(lab, g), (f'not {lab}', ~g)]:
        if y[m_].nunique() == 2:
            say(f'  AUC within {c} = {name}: {roc_auc_score(y[m_], pa[m_]):.3f}  (n = {int(m_.sum())})')

covX = cv[['age', 'PMI', 'RIN', 'APOE4']].fillna(cv.median()).assign(sex=(meta['Sex'] == 'Male').astype(float))
covX = StandardScaler().fit_transform(covX)
lp = np.log(np.clip(pa, 1e-4, 1 - 1e-4) / (1 - np.clip(pa, 1e-4, 1 - 1e-4))).to_numpy()[:, None]


def loglik(X):
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


lr_stat = 2 * (loglik(np.hstack([covX, lp])) - loglik(covX))
say(f'Likelihood-ratio test, label ~ covariates + logit(P_A) vs label ~ covariates: '
    f'chi2 = {lr_stat:.2f}, p = {chi2.sf(lr_stat, 1):.4f}  (does expression add beyond covariates?)')

# 6 ----------------------------------------------------------------------------------------------
say('\n=== 6. Stability ===')
hv = Parallel(n_jobs=N_JOBS)(delayed(lambda n: (n, roc_auc_score(y, model_a(fold0, n))))(n)
                             for n in [500, 1000, 2000, 5000])
say('AUC vs number of genes per subclass model (sensitivity only; 2000 was fixed in advance): '
    + ', '.join(f'{n}: {a:.3f}' for n, a in hv))
for sc in ['L4 IT', 'L2/3 IT']:
    sets = diag[diag.subclass == sc].top_genes.tolist()
    jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    counts = pd.Series([g for s in sets for g in s]).value_counts()
    say(f'{sc}: top-50 genes, mean Jaccard overlap between folds {np.mean(jac):.2f}; '
        f'in all 5 folds: {", ".join(counts[counts == 5].index[:25]) or "none"}')

open(f'{OUT}/04c_checks.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(f'\nSaved {OUT}/04c_checks.txt')
