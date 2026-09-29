#!/usr/bin/env python
"""
v3 Step 5d (CPU): pool the 5 fine-tuning folds (05c) into out-of-fold donor predictions for the 67 development
donors and compare with the baselines on the same donors. Locked test donors are not involved.

Compared: A (gene pseudobulk, all neurons; the bar), A_sub500 (genes, same 500 cells), E_sub_avg (frozen scGPT),
FT (fine-tuned scGPT; mean cell probability per donor). If several seeds exist, FT = mean over seeds.

Reported: AUC [bootstrap CI], accuracy, balanced accuracy, AUC in non-demented donors, Spearman with Braak 0-VI,
paired-bootstrap AUC difference vs each baseline, within-label confounder check, and per-fold training curves
(train loss vs validation loss and AUC - the overfitting evidence). No permutation test: it would need hundreds
of re-trainings; the paired comparison on identical donors is the evidence.

Inputs:  seaad/results/05c_fold*_seed*.json, 05b_oof_predictions.csv, 01_donors.csv, 02_splits.csv
Output:  seaad/results/05d_finetune_eval.txt, 05d_oof_predictions.csv

Usage:   python seaad/05d_finetune_eval.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
OUT = 'seaad/results'
SEED, N_BOOT = 42, 2000

splits = pd.read_csv(f'{OUT}/02_splits.csv', index_col='Donor ID')
dev = splits[splits['split'] != 'test']
y = dev['label'].astype(int)
braak_num, dementia = dev['braak_num'].astype(int), dev['dementia'].astype(int)
meta = pd.read_csv(f'{OUT}/01_donors.csv', index_col='Donor ID').loc[y.index]

runs = [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}/05c_fold*_seed*.json'))]
folds_done = sorted({r['fold'] for r in runs})
assert folds_done == [0, 1, 2, 3, 4], f'fine-tuning folds present: {folds_done} (need 0-4)'
per_seed = {}
for r in runs:
    assert not set(r['test_donor_prob']) & set(splits.index[splits['split'] == 'test']), 'locked test donor found!'
    per_seed.setdefault(r['seed'], {}).update(r['test_donor_prob'])
ft = pd.DataFrame(per_seed).reindex(y.index)
assert ft.notna().all().all(), 'some development donors have no fine-tuned prediction'

cand = pd.read_csv(f'{OUT}/05b_oof_predictions.csv', index_col=0).loc[y.index, ['A', 'A_sub500', 'E_sub_avg']]
cand['FT'] = ft.mean(axis=1)

rng = np.random.default_rng(SEED)
BOOT = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)) if y.iloc[i].nunique() == 2]
boot = {k: np.array([roc_auc_score(y.iloc[i], cand[k].iloc[i]) for i in BOOT]) for k in cand}
nd = dementia == 0
rows = {}
for k in cand:
    p = cand[k]
    rows[k] = {'auc': roc_auc_score(y, p), 'ci_low': np.percentile(boot[k], 2.5),
               'ci_high': np.percentile(boot[k], 97.5), 'accuracy': ((p >= 0.5).astype(int) == y).mean(),
               'balanced_acc': balanced_accuracy_score(y, (p >= 0.5).astype(int)),
               'auc_non_demented': roc_auc_score(y[nd], p[nd]), 'spearman_braak': spearmanr(p, braak_num).correlation}
res = pd.DataFrame(rows).T
diffs = pd.DataFrame({f'FT_vs_{b}': [f'{res.loc["FT", "auc"] - res.loc[b, "auc"]:+.3f} '
                                     f'[{np.percentile(boot["FT"] - boot[b], 2.5):+.3f}, '
                                     f'{np.percentile(boot["FT"] - boot[b], 97.5):+.3f}]']
                      for b in ['A', 'A_sub500', 'E_sub_avg']}, index=['dAUC [95% CI]']).T

cvars = {'neurons': meta['n_neurons_used'], 'RIN': pd.to_numeric(meta['RIN'], errors='coerce'),
         'PMI': pd.to_numeric(meta['PMI'], errors='coerce')}
conf = pd.DataFrame({v: {g: spearmanr(cand['FT'][(y == i) & s.notna()], s[(y == i) & s.notna()]).correlation
                         for i, g in [(0, 'within 0-IV'), (1, 'within V-VI')]} for v, s in cvars.items()}).T

curves = pd.DataFrame([{
    'fold': r['fold'], 'seed': r['seed'], 'epochs': r['config']['epochs_trained'],
    'best_epoch': r['config']['best_epoch'],
    'train_loss_first->best': f'{r["history"]["train_loss"][0]:.3f} -> '
                              f'{r["history"]["train_loss"][r["config"]["best_epoch"] - 1]:.3f}',
    'val_loss_first->best': f'{r["history"]["val_cell_loss"][0]:.3f} -> '
                            f'{r["history"]["val_cell_loss"][r["config"]["best_epoch"] - 1]:.3f}',
    'val_donor_auc_best': max(r['history']['val_donor_auc']),
    'test_donor_auc_fold': r['test_donor_auc_this_fold']} for r in runs]).set_index(['fold', 'seed'])

cand.join(dev[['split', 'label', 'braak_num', 'dementia']]).to_csv(f'{OUT}/05d_oof_predictions.csv')
txt = '\n'.join([
    f'Development donors: {len(y)} (locked test untouched). Fine-tuned scGPT: {len(runs)} runs, seeds {sorted(per_seed)}.',
    '', res.astype(float).round(3).to_string(), '',
    'Paired-bootstrap AUC difference, fine-tuned scGPT minus baseline:', diffs.to_string(), '',
    'Confounders - Spearman of FT probability with donor variables within each label group:',
    conf.round(3).to_string(), '',
    'Training curves per fold (train loss falling while validation loss rises = overfitting to training donors):',
    curves.round(3).to_string(),
])
open(f'{OUT}/05d_finetune_eval.txt', 'w', encoding='utf-8').write(txt + '\n')
print(txt)
