#!/usr/bin/env python
"""
v3 Step 4: donor-level baselines on the 67 DEVELOPMENT donors (5 donor folds).
The locked test donors are dropped on load and never touched here.

Target: Braak V-VI (1) vs 0-IV (0). One sample = one donor.

Models (everything - gene filter, HVGs, scaling, C - is fit on the training donors of each fold):
  majority        predicts the training-fold base rate
  covariates      age, sex, APOE4 dose, PMI, RIN                      -> is Braak predictable without expression?
  composition     neuron subclass proportions per donor (CLR)          -> is neuron loss alone enough?
  pseudobulk      all-neuron pseudobulk, log-CPM, top 2000 HVGs        -> main expression baseline
  subclass_avg    one pseudobulk model per major neuron subclass, probabilities averaged
  sub:<name>      the per-subclass models on their own                 -> which neurons carry the signal

Reported per model (out-of-fold predictions pooled over the 67 donors):
  AUC with donor-bootstrap 95% CI, balanced accuracy at 0.5, AUC within non-demented donors,
  Spearman(predicted probability, Braak 0-VI), mean +/- SD of per-fold AUC.
  Headline models (covariates, composition, pseudobulk, subclass_avg) also get a label-permutation
  p-value: the whole CV is rerun N_PERM times with shuffled labels. A synthetic null run gave AUCs of
  0.36-0.63 at n = 67, so an AUC near 0.6 alone is not evidence. sub:<name> models are exploratory
  (8 comparisons, no correction).

Inputs:  data/seaad_pseudobulk_donor.h5ad, data/seaad_pseudobulk_subclass.h5ad (03_extract_neurons.py)
         seaad/results/01_donors.csv
Outputs: seaad/results/04_baselines.csv, 04_baselines.txt, 04_oof_predictions.csv

Usage:   python seaad/04_baselines.py      (N_PERM env var, default 200; uses SLURM_CPUS_PER_TASK cores)
"""
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ.setdefault('PYTHONWARNINGS', 'ignore::FutureWarning')   # also silences joblib workers
PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
OUT = 'seaad/results'
SEED = 42
N_HVG = 2000
MIN_CPM = 10            # gene must average >= 10 CPM in the training donors
N_BOOT = 2000
N_PERM = int(os.environ.get('N_PERM', 200))
N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
MAJOR_SUBCLASSES = ['L2/3 IT', 'L4 IT', 'L5 IT', 'L6 IT', 'Vip', 'Pvalb', 'Sst', 'Lamp5']

# ---------------------------------------------------------------- data (development donors only)
pbd = ad.read_h5ad('data/seaad_pseudobulk_donor.h5ad')
pbs = ad.read_h5ad('data/seaad_pseudobulk_subclass.h5ad')
meta = pd.read_csv(f'{OUT}/01_donors.csv', index_col='Donor ID')

dev = pbd.obs.index[pbd.obs['split'] != 'test']
pbd = pbd[dev].copy()
pbs = pbs[pbs.obs['Donor ID'].isin(dev)].copy()
assert not (pbd.obs['split'] == 'test').any() and not (pbs.obs['split'] == 'test').any()

y = pbd.obs['label'].astype(int)
fold = pbd.obs['split'].str.replace('fold', '').astype(int)
braak_num = pbd.obs['braak_num'].astype(int)
dementia = pbd.obs['dementia'].astype(int)
print(f'Development donors: {len(dev)}  (V-VI: {int(y.sum())}, 0-IV: {int((1 - y).sum())}); '
      f'folds: {sorted(fold.unique())}')


def log_cpm(X):
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e6)


class ExprFeatures:
    """Gene filter + top-variance genes, chosen on training donors only."""
    def fit(self, X):
        cpm = np.expm1(X)
        expressed = cpm.mean(axis=0) >= MIN_CPM
        var = np.where(expressed, X.var(axis=0), -np.inf)
        self.idx = np.argsort(var)[::-1][:min(N_HVG, int(expressed.sum()))]
        return self

    def transform(self, X):
        return X[:, self.idx]


def lr():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 13), cv=3, penalty='l2', scoring='roc_auc', class_weight='balanced',
        max_iter=5000, random_state=SEED))


def run_cv(X, yv, expr=False):
    """Out-of-fold P(V-VI) for each donor. X: donors x features, aligned to the label Series yv."""
    folds = fold[yv.index]
    oof = pd.Series(np.nan, index=yv.index)
    for k in sorted(folds.unique()):
        tr, te = (folds != k).to_numpy(), (folds == k).to_numpy()
        Xtr, Xte = X[tr], X[te]
        if expr:
            f = ExprFeatures().fit(Xtr)
            Xtr, Xte = f.transform(Xtr), f.transform(Xte)
        m = lr().fit(Xtr, yv[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def majority_cv():
    oof = pd.Series(np.nan, index=y.index)
    for k in sorted(fold.unique()):
        oof[fold == k] = y[fold != k].mean()
    return oof


def summarize(name, p):
    ok = p.notna()
    p, yy = p[ok], y[ok]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(p), len(p))
        if yy.iloc[i].nunique() == 2:
            boots.append(roc_auc_score(yy.iloc[i], p.iloc[i]))
    nd = (dementia[ok] == 0)
    per_fold = [roc_auc_score(yy[fold[ok] == k], p[fold[ok] == k])
                for k in sorted(fold.unique()) if yy[fold[ok] == k].nunique() == 2]
    return {
        'model': name, 'n_donors': int(ok.sum()),
        'auc': roc_auc_score(yy, p) if p.nunique() > 1 else 0.5,
        'auc_ci_low': np.percentile(boots, 2.5), 'auc_ci_high': np.percentile(boots, 97.5),
        'balanced_acc': balanced_accuracy_score(yy, (p >= 0.5).astype(int)),
        'auc_non_demented': roc_auc_score(yy[nd], p[nd]) if p[nd].nunique() > 1 else 0.5,
        'spearman_braak': spearmanr(p, braak_num[ok]).correlation if p.nunique() > 1 else 0.0,
        'fold_auc_mean': np.mean(per_fold), 'fold_auc_sd': np.std(per_fold, ddof=1),
    }


preds = {'majority': majority_cv()}

# ---------------------------------------------------------------- covariates
cov = meta.loc[y.index, ['Age at Death', 'Sex', 'APOE Genotype', 'PMI', 'RIN']].copy()
cov['Sex'] = (cov['Sex'] == 'Male').astype(float)
cov['APOE4'] = cov['APOE Genotype'].astype(str).str.count('4').astype(float)
cov = cov.drop(columns='APOE Genotype').apply(pd.to_numeric, errors='coerce')
missing = cov.isna().sum()
cov = cov.fillna(cov.median())      # label-free imputation; counts reported below
X_cov = cov.to_numpy()
preds['covariates'] = run_cv(X_cov, y)

# ---------------------------------------------------------------- neuron composition (CLR of proportions)
comp = (pbs.obs.pivot_table(index='Donor ID', columns='Subclass', values='n_cells',
                            aggfunc='sum', fill_value=0, observed=True)
        .reindex(y.index).fillna(0))
lp = np.log(comp + 0.5)
X_comp = lp.sub(lp.mean(axis=1), axis=0).to_numpy()
preds['composition'] = run_cv(X_comp, y)

# ---------------------------------------------------------------- all-neuron pseudobulk
X_pb = log_cpm(pbd.X)
preds['pseudobulk'] = run_cv(X_pb, y, expr=True)

# ---------------------------------------------------------------- per-subclass pseudobulk
sub_X = {}
for sc in MAJOR_SUBCLASSES:
    s_ = pbs[pbs.obs['Subclass'].astype(str) == sc]
    s_ = s_[s_.obs['n_cells'] >= 20]        # too few cells -> unreliable pseudobulk; donor skipped
    Xs = pd.DataFrame(log_cpm(s_.X), index=s_.obs['Donor ID'].to_numpy()).reindex(y.index)
    have = Xs.notna().all(axis=1)
    sub_X[sc] = (Xs[have].to_numpy(), have)
    print(f'  {sc}: {int(have.sum())} donors with >= 20 cells')


def subclass_cv(yv):
    return {sc: run_cv(Xh, yv[have], expr=True).reindex(yv.index) for sc, (Xh, have) in sub_X.items()}


sub_preds = subclass_cv(y)
for sc, p in sub_preds.items():
    preds[f'sub:{sc}'] = p
preds['subclass_avg'] = pd.DataFrame(sub_preds).mean(axis=1)

# ---------------------------------------------------------------- permutation test (headline models)
HEADLINE = {
    'covariates': lambda yv: run_cv(X_cov, yv),
    'composition': lambda yv: run_cv(X_comp, yv),
    'pseudobulk': lambda yv: run_cv(X_pb, yv, expr=True),
    'subclass_avg': lambda yv: pd.DataFrame(subclass_cv(yv)).mean(axis=1),
}


def null_aucs(yp):
    return {name: roc_auc_score(yp, fn(yp)) for name, fn in HEADLINE.items()}


prng = np.random.default_rng(SEED)
perms = [pd.Series(prng.permutation(y.to_numpy()), index=y.index) for _ in range(N_PERM)]
print(f'Permutation test: {N_PERM} permutations on {N_JOBS} cores ...', flush=True)
null = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(null_aucs)(yp) for yp in perms))
perm_p, null_auc = {}, {}
for name in HEADLINE:
    real = roc_auc_score(y, preds[name])
    perm_p[name] = (1 + (null[name] >= real).sum()) / (1 + N_PERM)
    null_auc[name] = (np.percentile(null[name], 2.5), np.percentile(null[name], 97.5))
    print(f'  {name}: AUC {real:.3f}, null 95% [{null_auc[name][0]:.3f}-{null_auc[name][1]:.3f}], '
          f'p = {perm_p[name]:.4f}')

# ---------------------------------------------------------------- report
order = ['majority', 'covariates', 'composition', 'pseudobulk', 'subclass_avg'] + \
        [f'sub:{s}' for s in MAJOR_SUBCLASSES]
res = pd.DataFrame([summarize(n, preds[n]) for n in order]).set_index('model')
res['perm_p'] = pd.Series(perm_p)
res['null_auc_95'] = pd.Series({k: f'{a:.3f}-{b:.3f}' for k, (a, b) in null_auc.items()})
res.round(4).to_csv(f'{OUT}/04_baselines.csv')
oof = pd.DataFrame(preds).join(pbd.obs[['split', 'label', 'braak_num', 'dementia']])
oof.to_csv(f'{OUT}/04_oof_predictions.csv')

fmt = res.copy()
fmt['AUC [95% CI]'] = [f'{a:.3f} [{l:.3f}-{h:.3f}]' for a, l, h in
                       zip(res.auc, res.auc_ci_low, res.auc_ci_high)]
fmt['fold AUC'] = [f'{m:.3f} ± {s:.3f}' for m, s in zip(res.fold_auc_mean, res.fold_auc_sd)]
table = fmt[['n_donors', 'AUC [95% CI]', 'fold AUC', 'balanced_acc', 'auc_non_demented',
             'spearman_braak', 'perm_p', 'null_auc_95']].round(3).fillna('').to_string()
with open(f'{OUT}/04_baselines.txt', 'w', encoding='utf-8') as fh:
    txt = (f'Development donors only (locked test untouched). Target: Braak V-VI vs 0-IV.\n'
           f'Covariate missing values imputed with the median: {missing[missing > 0].to_dict()}\n\n'
           f'{table}\n\nauc_non_demented: AUC within the {int((dementia == 0).sum())} non-demented donors\n'
           f'spearman_braak: Spearman(P(V-VI), Braak 0-VI)\n'
           f'perm_p: label-permutation p ({N_PERM} permutations of the whole CV); '
           f'null_auc_95: 95% range of AUC under shuffled labels\n'
           f'sub:<name> rows are exploratory (8 subclasses, no multiple-comparison correction)\n')
    fh.write(txt)
print('\n' + txt)
