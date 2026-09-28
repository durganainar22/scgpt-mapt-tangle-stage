#!/usr/bin/env python
"""
v3 Step 5b (CPU): is the frozen scGPT representation better than raw genes for Braak V-VI vs 0-IV?
Development donors only (same 5 donor folds); the locked test donors are dropped on load.

Models (same logistic-regression recipe as model A, fit inside the training folds):
  E_sub_avg   per neuron subclass: donor-mean scGPT embedding -> LR; probabilities averaged over subclasses
  E_all       donor-mean scGPT embedding over all subsampled neurons -> LR
  A_sub500    FAIRNESS CONTROL: gene pseudobulk (as model A) but from the SAME 500 subsampled cells per donor
              that scGPT saw, so E_sub_avg vs A_sub500 compares representations, not cell numbers
  A           model A itself (all neurons; from 04b_oof_predictions.csv) - the pre-registered bar, 0.729

Reported: AUC [bootstrap CI], accuracy, balanced accuracy, AUC in non-demented donors, Spearman with
Braak 0-VI, paired-bootstrap AUC difference vs A and vs A_sub500, label-permutation p (whole CV rerun),
and the step-4c confounder check (Spearman of P with neurons / RIN / PMI within each label group).

Inputs:  data/seaad_scgpt_frozen_emb.npy, data/seaad_scgpt_frozen_obs.csv (05a), data/seaad_neurons_sub500.h5ad,
         seaad/results/01_donors.csv, 02_splits.csv, 04b_oof_predictions.csv
Output:  seaad/results/05b_embed_probe.txt, 05b_oof_predictions.csv

Usage:   python seaad/05b_embed_probe.py     (N_PERM env var, default 200; SLURM_CPUS_PER_TASK cores)
"""
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
OUT = 'seaad/results'
SEED, N_HVG, MIN_CPM, N_BOOT = 42, 2000, 10, 2000
MIN_CELLS = 10              # sampled cells of a subclass needed for a donor-mean (500 cells/donor in total)
MIN_DONORS = 30             # a subclass block is used only if this many donors reach MIN_CELLS
N_PERM = int(os.environ.get('N_PERM', 200))
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
NEURON_SUBCLASSES = ['L2/3 IT', 'L4 IT', 'L5 IT', 'L6 IT', 'Vip', 'Pvalb', 'Sst', 'Lamp5']

splits = pd.read_csv(f'{OUT}/02_splits.csv', index_col='Donor ID')
dev = splits[splits['split'] != 'test']
y = dev['label'].astype(int)
fold = dev['split'].str.replace('fold', '').astype(int)
braak_num, dementia = dev['braak_num'].astype(int), dev['dementia'].astype(int)
meta = pd.read_csv(f'{OUT}/01_donors.csv', index_col='Donor ID').loc[y.index]

emb = np.load('data/seaad_scgpt_frozen_emb.npy')
obs = pd.read_csv('data/seaad_scgpt_frozen_obs.csv', index_col=0)
assert len(obs) == len(emb)
keep = obs['Donor ID'].isin(y.index).to_numpy()
emb, obs = emb[keep], obs[keep]
assert not (obs['split'] == 'test').any()
cells = ad.read_h5ad('data/seaad_neurons_sub500.h5ad')
cells = cells[cells.obs['Donor ID'].isin(y.index)].copy()
assert (cells.obs_names == obs.index).all(), 'embedding rows and h5ad rows must match'


def log_cpm(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e6)


def top_genes(Xtr):
    expressed = np.expm1(Xtr).mean(axis=0) >= MIN_CPM
    var = np.where(expressed, Xtr.var(axis=0), -np.inf)
    return np.argsort(var)[::-1][:min(N_HVG, int(expressed.sum()))]


def lr():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 13), cv=3, penalty='l2', scoring='roc_auc', class_weight='balanced',
        max_iter=5000, random_state=SEED))


def donor_table(mask, how):
    """donors x features for the cells in mask: mean embedding or summed-UMI log-CPM."""
    d = obs['Donor ID'].to_numpy()[mask]
    ids = pd.Index(sorted(set(d)))
    grp = sp.csr_matrix((np.ones(len(d)), (ids.get_indexer(d), np.arange(len(d)))), shape=(len(ids), len(d)))
    if how == 'emb':
        n = np.asarray(grp.sum(axis=1))
        X = (grp @ emb[mask]) / n
    else:
        X = log_cpm((grp @ cells.X[mask]).toarray())
    return pd.DataFrame(X, index=ids).reindex(y.index)


def cv_block(X, yv, expr):
    have = X.notna().all(axis=1)
    Xh, yh, fh = X[have].to_numpy(), yv[have], fold[have]
    p = pd.Series(np.nan, index=yv.index)
    for k in sorted(fh.unique()):
        tr, te = (fh != k).to_numpy(), (fh == k).to_numpy()
        if yh[tr].nunique() < 2:
            continue
        Xtr, Xte = Xh[tr], Xh[te]
        if expr:
            g = top_genes(Xtr)
            Xtr, Xte = Xtr[:, g], Xte[:, g]
        p[have[have].index[te]] = lr().fit(Xtr, yh[tr]).predict_proba(Xte)[:, 1]
    return p


sub = obs['Subclass'].astype(str).to_numpy()
tables = {'E_all': (donor_table(np.ones(len(obs), bool), 'emb'), False)}
USED = []
for s in NEURON_SUBCLASSES:
    m = sub == s
    n_per = pd.Series(obs['Donor ID'].to_numpy()[m]).value_counts()
    n_ok = int((n_per >= MIN_CELLS).sum())
    print(f'  {s}: {n_ok} donors with >= {MIN_CELLS} sampled cells'
          + ('' if n_ok >= MIN_DONORS else f' -> skipped (< {MIN_DONORS} donors)'), flush=True)
    if n_ok < MIN_DONORS:
        continue
    ok = np.isin(obs['Donor ID'].to_numpy(), n_per[n_per >= MIN_CELLS].index) & m
    tables[f'E:{s}'] = (donor_table(ok, 'emb'), False)
    tables[f'G:{s}'] = (donor_table(ok, 'genes'), True)
    USED.append(s)


def all_models(yv):
    p = {k: cv_block(X, yv, expr) for k, (X, expr) in tables.items()}
    return {'E_sub_avg': pd.DataFrame({s: p[f'E:{s}'] for s in USED}).mean(axis=1),
            'E_all': p['E_all'],
            'A_sub500': pd.DataFrame({s: p[f'G:{s}'] for s in USED}).mean(axis=1)}, p


cand, per_block = all_models(y)
cand['A'] = pd.read_csv(f'{OUT}/04b_oof_predictions.csv', index_col=0).loc[y.index, 'A neuron_avg']

rng = np.random.default_rng(SEED)
BOOT = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)) if y.iloc[i].nunique() == 2]
boot = {k: np.array([roc_auc_score(y.iloc[i], v.iloc[i]) for i in BOOT]) for k, v in cand.items()}


def diff(a, b):
    d = boot[a] - boot[b]
    return (f'{roc_auc_score(y, cand[a]) - roc_auc_score(y, cand[b]):+.3f} '
            f'[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')


prng = np.random.default_rng(SEED)
perms = [pd.Series(prng.permutation(y.to_numpy()), index=y.index) for _ in range(N_PERM)]
print(f'Permutation test: {N_PERM} permutations on {N_JOBS} cores ...', flush=True)
null = pd.DataFrame(Parallel(n_jobs=N_JOBS)(
    delayed(lambda yp: {k: roc_auc_score(yp, v) for k, v in all_models(yp)[0].items()})(yp) for yp in perms))

nd = dementia == 0
rows = {}
for k in ['A', 'A_sub500', 'E_sub_avg', 'E_all']:
    p = cand[k]
    rows[k] = {'auc': roc_auc_score(y, p), 'ci_low': np.percentile(boot[k], 2.5),
               'ci_high': np.percentile(boot[k], 97.5), 'accuracy': ((p >= 0.5).astype(int) == y).mean(),
               'balanced_acc': balanced_accuracy_score(y, (p >= 0.5).astype(int)),
               'auc_non_demented': roc_auc_score(y[nd], p[nd]),
               'spearman_braak': spearmanr(p, braak_num).correlation,
               'perm_p': (1 + (null[k] >= roc_auc_score(y, p)).sum()) / (1 + N_PERM) if k in null else np.nan,
               'dAUC_vs_A': diff(k, 'A'), 'dAUC_vs_A_sub500': diff(k, 'A_sub500')}
res = pd.DataFrame(rows).T

conf = []
cvars = {'neurons': meta['n_neurons_used'], 'RIN': pd.to_numeric(meta['RIN'], errors='coerce'),
         'PMI': pd.to_numeric(meta['PMI'], errors='coerce')}
for k in ['A', 'E_sub_avg']:
    for v, s in cvars.items():
        conf.append({'model': k, 'variable': v,
                     **{f'rho_within_{g}': spearmanr(cand[k][(y == i) & s.notna()], s[(y == i) & s.notna()]).correlation
                        for i, g in [(0, '0-IV'), (1, 'V-VI')]}})

pd.DataFrame(cand).join(dev[['split', 'label', 'braak_num', 'dementia']]).to_csv(f'{OUT}/05b_oof_predictions.csv')
num = ['auc', 'ci_low', 'ci_high', 'accuracy', 'balanced_acc', 'auc_non_demented', 'spearman_braak', 'perm_p']
res[num] = res[num].astype(float)
txt = '\n'.join([
    f'Development donors: {len(y)} (locked test untouched). Target: Braak V-VI vs 0-IV. Frozen pretrained scGPT.',
    f'Neuron subclasses used (>= {MIN_DONORS} donors with >= {MIN_CELLS} sampled cells): {USED}',
    '', res[num].round(3).fillna('').to_string(), '',
    'Paired-bootstrap AUC difference [95% CI]:', res[['dAUC_vs_A', 'dAUC_vs_A_sub500']].to_string(), '',
    'Per-subclass AUC (E = scGPT embedding, G = genes, same sampled cells): ' + ', '.join(
        f'{k} {roc_auc_score(y[v.notna()], v.dropna()):.3f}' for k, v in per_block.items()
        if v.notna().sum() and y[v.notna()].nunique() == 2), '',
    'Confounders - Spearman of P with donor variables within each label group:',
    pd.DataFrame(conf).set_index(['model', 'variable']).round(3).to_string(),
])
open(f'{OUT}/05b_embed_probe.txt', 'w', encoding='utf-8').write(txt + '\n')
print('\n' + txt)
