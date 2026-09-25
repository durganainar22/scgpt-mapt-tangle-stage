#!/usr/bin/env python
"""
Combine per-fold results into one summary (mean +/- SD across donor folds/seeds).

Inputs:  results/scgpt_fold*_seed*.json  (scgpt_finetune.ipynb, one per fold/seed)
         results/baselines.json          (baselines.py)
Outputs: results_summary.json, results/summary.md

Usage:   python aggregate_results.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR', os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)

METRICS = ['balanced_accuracy', 'macro_f1', 'macro_auc', 'accuracy', 'donor_accuracy']
rows = []


def add(model, fold, seed, m):
    row = {'model': model, 'fold': fold, 'seed': seed}
    for k in METRICS[:-1]:
        row[k] = m.get(k)
    row['donor_accuracy'] = m.get('donor_level', {}).get('accuracy')
    for stage, r in m['per_class_recall'].items():
        row[f'recall_{stage}'] = r
    rows.append(row)


if os.path.exists('results/baselines.json'):
    base = json.load(open('results/baselines.json'))
    for f in base['folds']:
        add('majority', f['fold'], None, f['majority'])
        add('pca_logreg', f['fold'], None, f['pca_logreg'])

scgpt_files = sorted(glob.glob('results/scgpt_fold*_seed*.json'))
for path in scgpt_files:
    r = json.load(open(path))
    add('frozen_scgpt_probe', r['fold'], r['seed'], r['frozen_scgpt_probe'])
    add('finetuned_scgpt', r['fold'], r['seed'], r['finetuned_scgpt'])

if not rows:
    raise SystemExit('No results found in results/ - run baselines.py and the finetune notebook first.')

df = pd.DataFrame(rows)
order = ['majority', 'pca_logreg', 'frozen_scgpt_probe', 'finetuned_scgpt']
cols = METRICS + [c for c in df.columns if c.startswith('recall_')]
summary = (df.groupby('model')[cols]
           .agg(['mean', 'std', 'count'])
           .reindex([m for m in order if m in df['model'].unique()]))

fmt = pd.DataFrame(index=summary.index)
for c in cols:
    fmt[c] = [f"{m:.3f} ± {s:.3f}" if pd.notna(s) else (f"{m:.3f}" if pd.notna(m) else '-')
              for m, s in zip(summary[(c, 'mean')], summary[(c, 'std')])]
fmt['n_runs'] = summary[(cols[0], 'count')].astype(int)

print(fmt[METRICS + ['n_runs']].to_string())

out = {
    'task': 'Tangle stage classification (Stage 1/2/5/6), neurons, GSE174367',
    'evaluation': 'donor-level cross-validation (test donors never seen in training); mean ± SD across folds/seeds',
    'n_scgpt_runs': len(scgpt_files),
    'summary': {model: {c: {'mean': summary.loc[model, (c, 'mean')],
                            'std': summary.loc[model, (c, 'std')]}
                        for c in cols}
                for model in summary.index},
    'per_run': df.to_dict(orient='records'),
}
with open('results_summary.json', 'w') as f:
    json.dump(out, f, indent=2, default=lambda x: None if pd.isna(x) else float(x))

os.makedirs('results', exist_ok=True)
with open('results/summary.md', 'w') as f:
    f.write('| Model | ' + ' | '.join(METRICS) + ' | runs |\n')
    f.write('|' + '---|' * (len(METRICS) + 2) + '\n')
    for model, r in fmt.iterrows():
        f.write(f'| {model} | ' + ' | '.join(r[METRICS]) + f" | {r['n_runs']} |\n")
print('\nSaved results_summary.json and results/summary.md')
