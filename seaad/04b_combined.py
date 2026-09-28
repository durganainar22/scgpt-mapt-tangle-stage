#!/usr/bin/env python
"""
v3 Step 4b: can more information push Braak V-VI vs 0-IV further? Development donors only (5 donor folds);
the locked test donors are dropped on load.

Feature blocks (each is its own L2 logistic model, fit inside the training folds as in step 4):
  cov            age, sex, APOE4 dose, PMI, RIN
  supertype      neuron SUPERTYPE composition (CLR of proportions, 117 types; from the metadata CSV)
  n:<subclass>   pseudobulk of 8 major neuron subclasses          (as in step 4)
  g:<subclass>   pseudobulk of each non-neuronal subclass with >= 50 donors having >= 20 cells (03b)

Candidates, fixed before running (selection rule below):
  A neuron_avg    mean P over the 8 neuron-subclass models         (= step 4 subclass_avg, reproduced)
  B expr_avg      mean P over neuron + glia subclass models
  C all_avg       mean P over every block (expression + supertype + covariates)
  D stacked       logistic meta-model on the block probabilities; the meta-model is trained on INNER
                  out-of-fold predictions within each outer training set (nested), so no outer-test donor
                  influences it

SELECTION RULE (pre-registered): the final model for the locked test is the candidate with the highest
development AUC; if another candidate is within 0.01 AUC, the simpler one wins (order A < B < C < D).

Reported: AUC [bootstrap CI], accuracy and balanced accuracy at P = 0.5, accuracy per Braak stage,
AUC in non-demented donors, Spearman with Braak 0-VI, paired-bootstrap AUC difference vs covariates and
vs A, label-permutation p for B, C, D (whole nested CV rerun).

Inputs:  data/seaad_pseudobulk_subclass.h5ad (03), data/seaad_pseudobulk_glia.h5ad (03b),
         data/SEAAD_MTG_RNAseq_final-nuclei_metadata.2026-06-22.csv, seaad/results/01_donors.csv, 02_splits.csv
Outputs: seaad/results/04b_combined.txt, 04b_combined.csv, 04b_oof_predictions.csv

Usage:   python seaad/04b_combined.py      (N_PERM env var, default 100; uses SLURM_CPUS_PER_TASK cores)
"""
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
OUT = 'seaad/results'
META = 'data/SEAAD_MTG_RNAseq_final-nuclei_metadata.2026-06-22.csv'
SEED = 42
N_HVG, MIN_CPM, MIN_CELLS, MIN_DONORS_GLIA = 2000, 10, 20, 50
N_BOOT = 2000
N_PERM = int(os.environ.get('N_PERM', 100))
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
NEURON_SUBCLASSES = ['L2/3 IT', 'L4 IT', 'L5 IT', 'L6 IT', 'Vip', 'Pvalb', 'Sst', 'Lamp5']

# ---------------------------------------------------------------- development donors
splits = pd.read_csv(f'{OUT}/02_splits.csv', index_col='Donor ID')
dev = splits[splits['split'] != 'test']
y = dev['label'].astype(int)
fold = dev['split'].str.replace('fold', '').astype(int)
braak_num, dementia = dev['braak_num'].astype(int), dev['dementia'].astype(int)


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


class Block:
    """One feature block: X for the donors that have it (have = boolean mask over y.index)."""
    def __init__(self, X, have, expr):
        self.X, self.have, self.expr = X, have, expr
        self.pos = pd.Series(np.arange(int(have.sum())), index=have.index[have])

    def fit_predict(self, train_ids, test_ids, yv):
        tr = [d for d in train_ids if d in self.pos.index]
        te = [d for d in test_ids if d in self.pos.index]
        out = pd.Series(np.nan, index=test_ids)
        if not te or yv[tr].nunique() < 2:
            return out
        Xtr, Xte = self.X[self.pos[tr]], self.X[self.pos[te]]
        if self.expr:
            g = top_genes(Xtr)
            Xtr, Xte = Xtr[:, g], Xte[:, g]
        out[te] = lr().fit(Xtr, yv[tr]).predict_proba(Xte)[:, 1]
        return out


def pb_block(pb, sc):
    s = pb[(pb.obs['Subclass'].astype(str) == sc) & (pb.obs['n_cells'] >= MIN_CELLS)]
    Xs = pd.DataFrame(log_cpm(s.X), index=s.obs['Donor ID'].to_numpy()).reindex(y.index)
    have = Xs.notna().all(axis=1)
    return Block(Xs[have].to_numpy(), have, expr=True)


blocks = {}
# covariates
meta_d = pd.read_csv(f'{OUT}/01_donors.csv', index_col='Donor ID').loc[y.index]
cov = meta_d[['Age at Death', 'PMI', 'RIN']].apply(pd.to_numeric, errors='coerce')
cov['Sex'] = (meta_d['Sex'] == 'Male').astype(float)
cov['APOE4'] = meta_d['APOE Genotype'].astype(str).str.count('4').astype(float)
cov = cov.fillna(cov.median())
blocks['cov'] = Block(cov.to_numpy(), pd.Series(True, index=y.index), expr=False)
# neuron supertype composition
cells = pd.read_csv(META, usecols=['Donor ID', 'Class', 'Supertype', 'Used in analysis'])
cells = cells[cells['Donor ID'].isin(y.index) & cells['Used in analysis'].astype(bool)
              & cells['Class'].astype(str).str.startswith('Neuronal')]
tab = pd.crosstab(cells['Donor ID'], cells['Supertype']).reindex(y.index).fillna(0)
lp = np.log(tab.div(tab.sum(axis=1), axis=0) + 1e-4)
blocks['supertype'] = Block(lp.sub(lp.mean(axis=1), axis=0).to_numpy(), pd.Series(True, index=y.index), False)
del cells
# neuron and glia subclass pseudobulk
pbn = ad.read_h5ad('data/seaad_pseudobulk_subclass.h5ad')
pbg = ad.read_h5ad('data/seaad_pseudobulk_glia.h5ad')
for sc in NEURON_SUBCLASSES:
    blocks[f'n:{sc}'] = pb_block(pbn, sc)
glia_ok = pbg.obs[(pbg.obs['n_cells'] >= MIN_CELLS) & pbg.obs['Donor ID'].isin(y.index)] \
    .groupby('Subclass', observed=True).size()
GLIA = sorted(glia_ok[glia_ok >= MIN_DONORS_GLIA].index.astype(str))
for sc in GLIA:
    blocks[f'g:{sc}'] = pb_block(pbg, sc)
for name, b in blocks.items():
    print(f'  block {name}: {int(b.have.sum())} donors, {b.X.shape[1]} features', flush=True)

EXPR_N = [f'n:{s}' for s in NEURON_SUBCLASSES]
EXPR_G = [f'g:{s}' for s in GLIA]


# ---------------------------------------------------------------- CV machinery
def outer_block_preds(yv):
    """Outer-fold out-of-fold P for every block -> DataFrame donors x blocks."""
    out = pd.DataFrame(np.nan, index=yv.index, columns=list(blocks))
    for k in sorted(fold.unique()):
        tr, te = list(yv.index[fold != k]), list(yv.index[fold == k])
        for name, b in blocks.items():
            out.loc[te, name] = b.fit_predict(tr, te, yv).to_numpy()
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stacked_preds(yv, outer):
    """Nested stacking: meta-model trained on inner OOF block predictions within each outer training set."""
    res = pd.Series(np.nan, index=yv.index)
    for k in sorted(fold.unique()):
        tr, te = yv.index[fold != k], yv.index[fold == k]
        inner = pd.DataFrame(np.nan, index=tr, columns=list(blocks))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for itr, ite in skf.split(np.zeros(len(tr)), yv[tr]):
            a, b_ = list(tr[itr]), list(tr[ite])
            for name, blk in blocks.items():
                inner.loc[b_, name] = blk.fit_predict(a, b_, yv).to_numpy()
        meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=5000)
        meta.fit(logit(inner.fillna(0.5).to_numpy()), yv[tr])
        res[te] = meta.predict_proba(logit(outer.loc[te].fillna(0.5).to_numpy()))[:, 1]
    return res


def candidates(yv):
    outer = outer_block_preds(yv)
    return outer, {
        'A neuron_avg': outer[EXPR_N].mean(axis=1),
        'B expr_avg': outer[EXPR_N + EXPR_G].mean(axis=1),
        'C all_avg': outer.mean(axis=1),
        'D stacked': stacked_preds(yv, outer),
    }


outer, cand = candidates(y)


# ---------------------------------------------------------------- metrics
def boot_idx(n):
    rng = np.random.default_rng(SEED)
    return [rng.integers(0, n, n) for _ in range(N_BOOT)]


BOOT = [i for i in boot_idx(len(y)) if y.iloc[i].nunique() == 2]


def auc_boot(p):
    return np.array([roc_auc_score(y.iloc[i], p.iloc[i]) for i in BOOT])


def summarize(p, ref):
    b = auc_boot(p)
    nd = dementia == 0
    row = {'auc': roc_auc_score(y, p), 'ci_low': np.percentile(b, 2.5), 'ci_high': np.percentile(b, 97.5),
           'accuracy': ((p >= 0.5).astype(int) == y).mean(),
           'balanced_acc': balanced_accuracy_score(y, (p >= 0.5).astype(int)),
           'auc_non_demented': roc_auc_score(y[nd], p[nd]),
           'spearman_braak': spearmanr(p, braak_num).correlation}
    for rname, r in ref.items():
        d = b - r
        row[f'dAUC_vs_{rname}'] = f'{roc_auc_score(y, p) - roc_auc_score(y, ref_p[rname]):+.3f} ' \
                                  f'[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]'
    return row


ref_p = {'cov': outer['cov'], 'A': cand['A neuron_avg']}
ref_boot = {k: auc_boot(v) for k, v in ref_p.items()}
rows = {name: summarize(outer[name], ref_boot) for name in ['cov', 'supertype']}
rows.update({name: summarize(p, ref_boot) for name, p in cand.items()})
res = pd.DataFrame(rows).T

# ---------------------------------------------------------------- permutation test (B, C, D)
prng = np.random.default_rng(SEED)
perms = [pd.Series(prng.permutation(y.to_numpy()), index=y.index) for _ in range(N_PERM)]


def null_row(yp):
    _, c = candidates(yp)
    return {k: roc_auc_score(yp, v) for k, v in c.items() if k[0] in 'BCD'}


print(f'Permutation test: {N_PERM} permutations on {N_JOBS} cores ...', flush=True)
null = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(null_row)(yp) for yp in perms))
for k in null.columns:
    res.loc[k, 'perm_p'] = (1 + (null[k] >= res.loc[k, 'auc']).sum()) / (1 + N_PERM)

# ---------------------------------------------------------------- selection + report
order = ['A neuron_avg', 'B expr_avg', 'C all_avg', 'D stacked']
best = max(order, key=lambda k: res.loc[k, 'auc'])
chosen = next(k for k in order if res.loc[k, 'auc'] >= res.loc[best, 'auc'] - 0.01)

per_stage = pd.DataFrame({k: ((cand[k] >= 0.5).astype(int) == y).groupby(braak_num).mean() for k in order})
per_stage.insert(0, 'n', braak_num.value_counts().sort_index())

res.to_csv(f'{OUT}/04b_combined.csv')
pd.concat([outer, pd.DataFrame(cand)], axis=1).join(dev[['split', 'label', 'braak_num', 'dementia']]) \
    .to_csv(f'{OUT}/04b_oof_predictions.csv')

num = ['auc', 'ci_low', 'ci_high', 'accuracy', 'balanced_acc', 'auc_non_demented', 'spearman_braak', 'perm_p']
res[num] = res[num].astype(float)
txt = '\n'.join([
    f'Development donors: {len(y)} (locked test untouched). Target: Braak V-VI vs 0-IV.',
    f'Glia blocks (>= {MIN_DONORS_GLIA} donors with >= {MIN_CELLS} cells): {GLIA}', '',
    res[num].round(3).fillna('').to_string(), '',
    'Paired-bootstrap AUC difference [95% CI]:',
    res[['dAUC_vs_cov', 'dAUC_vs_A']].to_string(), '',
    'Accuracy by Braak stage (P >= 0.5):', per_stage.round(2).to_string(), '',
    f'Per-block AUC: ' + ', '.join(f'{c} {roc_auc_score(y[outer[c].notna()], outer[c].dropna()):.3f}'
                                   for c in outer.columns), '',
    f'SELECTED for locked test (pre-registered rule): {chosen}',
])
open(f'{OUT}/04b_combined.txt', 'w', encoding='utf-8').write(txt + '\n')
print('\n' + txt)
