#!/usr/bin/env python
"""
Non-deep-learning baselines on the SAME donor-level folds as scGPT.

  1. Majority class  - always predict the most common training stage
  2. PCA + logistic regression - PCA (50 comps) and scaling fit on training
     donors only, class_weight='balanced'

If fine-tuned scGPT cannot beat (2), the foundation model adds nothing here.
The frozen-scGPT-embedding probe needs a GPU, so it lives in scgpt_finetune.ipynb.

Usage:  python baselines.py            (run from the project directory)
Output: results/baselines.json
"""
import json
import os

import anndata as ad
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics_utils import compute_metrics, majority_baseline
from splits import make_donor_folds, split_indices, describe_split

PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR', os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
os.makedirs('results', exist_ok=True)

adata = ad.read_h5ad('neurons_hvg_preprocessed.h5ad')
obs = adata.obs
y = obs['tangle_label'].to_numpy().astype(int)
donors = obs['donor_id'].astype(str).to_numpy()
# X holds log1p-normalised expression (NOT scaled) after the preprocessing fix
X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
assert X.min() >= 0, 'X has negative values - it looks scaled; rerun scgpt_preprocessing.ipynb'

folds, n_splits = make_donor_folds(obs)
print(f'{n_splits} donor-level folds')

results = {'n_splits': n_splits, 'folds': []}
for k in range(n_splits):
    tr, va, te = split_indices(obs, folds, n_splits, test_fold=k)
    # baselines have no early stopping, so train on train+val
    tr = np.concatenate([tr, va])
    print(f'\n=== Fold {k} ===')
    print(describe_split(obs, {'train': tr, 'test': te}))

    maj = majority_baseline(y[tr], y[te], donors[te])

    lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=50, random_state=0),
        LogisticRegression(max_iter=2000, class_weight='balanced'),
    )
    lr.fit(X[tr], y[tr])
    probs = lr.predict_proba(X[te])
    pca_lr = compute_metrics(y[te], probs.argmax(1), probs, donors[te])

    for name, m in [('majority', maj), ('pca_logreg', pca_lr)]:
        print(f"  {name:<11} acc={m['accuracy']:.3f}  bal_acc={m['balanced_accuracy']:.3f}  "
              f"macro_f1={m['macro_f1']:.3f}  macro_auc={m['macro_auc']}")
    results['folds'].append({'fold': k, 'majority': maj, 'pca_logreg': pca_lr})

with open('results/baselines.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved results/baselines.json')
