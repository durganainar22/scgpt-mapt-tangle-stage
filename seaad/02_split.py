#!/usr/bin/env python
"""
v3 Step 2: donor-level split, fixed BEFORE any expression data is modelled.

Target (pre-registered): Braak V-VI (tangles in neocortex, i.e. in the MTG we sequenced) vs Braak 0-IV.
  - 0-IV = 35 donors, V-VI = 49 donors (see seaad/results/01_audit.txt)
  - secondary: Braak 0-VI as an ordinal number (Spearman)

Split:
  - LOCKED TEST: 17 donors (20%), stratified by Braak band (0-III/IV/V/VI) x dementia. Not used for any model choice;
    evaluated once, in the final benchmark step.
  - DEVELOPMENT: the other 67 donors, 5 donor folds for cross-validation, same strata.
    Every cell inherits its donor's split, so no donor is ever in two splits.

Input:   seaad/results/01_donors.csv
Outputs: seaad/results/02_splits.csv  (donor -> 'test' or fold 0-4, plus labels)
         seaad/results/02_splits.txt  (tables + SHA-256 of the test donor list)

Usage:   python seaad/02_split.py
"""
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
N_TEST = 17
N_FOLDS = 5
BRAAK_ORDINAL = {'Braak 0': 0, 'Braak I': 1, 'Braak II': 2, 'Braak III': 3,
                 'Braak IV': 4, 'Braak V': 5, 'Braak VI': 6}

PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
OUT = 'seaad/results'

donors = pd.read_csv(f'{OUT}/01_donors.csv', index_col='Donor ID').sort_index()
donors['braak_num'] = donors['Braak'].map(BRAAK_ORDINAL)
assert donors['braak_num'].notna().all(), donors.loc[donors['braak_num'].isna(), 'Braak']
donors['label'] = (donors['braak_num'] >= 5).astype(int)            # 1 = Braak V-VI
donors['dementia'] = (donors['Cognitive Status'] == 'Dementia').astype(int)
# Strata: Braak band (0-III / IV / V / VI) x dementia, so the test set spans the whole Braak range
# (stratifying on the binary label alone left the test set with no Braak 0-II and one Braak VI donor).
# Braak VI has a single non-demented donor, so Braak VI is one stratum.
band = pd.cut(donors['braak_num'], [-1, 3, 4, 5, 6], labels=['0-III', 'IV', 'V', 'VI']).astype(str)
strata = band + '_' + donors['dementia'].astype(str)
strata[band == 'VI'] = 'VI'

dev, test = train_test_split(donors.index, test_size=N_TEST, stratify=strata, random_state=SEED)
donors['split'] = ''
donors.loc[test, 'split'] = 'test'

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
dev = np.array(sorted(dev))
for k, (_, idx) in enumerate(skf.split(dev, strata[dev])):
    donors.loc[dev[idx], 'split'] = f'fold{k}'
assert (donors['split'] != '').all()

cols = ['split', 'label', 'braak_num', 'Braak', 'dementia', 'Cognitive Status', 'Sex',
        'Age at Death', 'APOE Genotype', 'n_neurons_used']
donors[cols].to_csv(f'{OUT}/02_splits.csv')

test_ids = sorted(donors.index[donors['split'] == 'test'])
digest = hashlib.sha256(','.join(test_ids).encode()).hexdigest()

with open(f'{OUT}/02_splits.txt', 'w', encoding='utf-8') as f:
    def say(*a):
        print(*a)
        print(*a, file=f)
    say(f'Donors: {len(donors)}  test: {len(test_ids)}  dev: {len(dev)} in {N_FOLDS} folds  (seed {SEED})')
    say('\n=== Donors per split x target (0 = Braak 0-IV, 1 = Braak V-VI) ===')
    say(pd.crosstab(donors['split'], donors['label'], margins=True).to_string())
    say('\n=== Donors per split x dementia ===')
    say(pd.crosstab(donors['split'], donors['dementia'], margins=True).to_string())
    say('\n=== Donors per split x Braak stage ===')
    say(pd.crosstab(donors['split'], donors['Braak']).to_string())
    say('\n=== Neurons per split ===')
    say(donors.groupby('split')['n_neurons_used'].agg(['sum', 'min', 'median']).to_string())
    say(f'\nLocked test donors: {test_ids}')
    say(f'SHA-256 of test donor list: {digest}')
