"""
Donor-level cross-validation folds shared by every model in this project.

Tangle stage is a per-donor label, so cells from the same donor must never be
split across train / val / test. Folds are assigned to DONORS (stratified by
the donor's stage) and cells inherit their donor's fold. The fold assignment
uses a fixed seed so the scGPT notebook and baselines.py see identical splits.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

FOLD_SEED = 42
DONOR_CANDIDATES = ['Sample.ID', 'SampleID', 'Sample', 'sample_id', 'sample',
                    'Donor', 'donor', 'donor_id', 'Individual', 'Subject']


def find_donor_column(columns):
    """Return the first metadata column that looks like a donor/sample ID."""
    for c in DONOR_CANDIDATES:
        if c in columns:
            return c
    raise KeyError(
        f'No donor/sample ID column found. Looked for {DONOR_CANDIDATES}. '
        f'Available columns: {list(columns)}'
    )


def donor_table(obs, donor_col='donor_id', label_col='tangle_label'):
    """One row per donor with its (single) label and number of cells."""
    per_donor = obs.groupby(donor_col, observed=True)[label_col].agg(['nunique', 'first', 'size'])
    bad = per_donor[per_donor['nunique'] > 1]
    if len(bad):
        raise ValueError(f'Donors with more than one label (unexpected): {bad.index.tolist()}')
    return per_donor.rename(columns={'first': 'label', 'size': 'n_cells'})[['label', 'n_cells']]


def make_donor_folds(obs, donor_col='donor_id', label_col='tangle_label', max_splits=5):
    """
    Assign each donor to a fold, stratified by stage.

    n_splits = min(max_splits, fewest donors in any stage), so every stage has
    at least one donor in every test fold.

    Returns
    -------
    folds : pd.Series  donor -> fold index
    n_splits : int
    """
    donors = donor_table(obs, donor_col, label_col)
    min_per_stage = donors['label'].value_counts().min()
    if min_per_stage < 2:
        raise ValueError(
            'At least one stage has only 1 donor, so it cannot appear in both '
            'train and test. Donors per stage:\n'
            f"{donors['label'].value_counts().sort_index()}"
        )
    n_splits = int(min(max_splits, min_per_stage))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=FOLD_SEED)
    folds = pd.Series(-1, index=donors.index, name='fold')
    for k, (_, test_idx) in enumerate(skf.split(np.zeros(len(donors)), donors['label'])):
        folds.iloc[test_idx] = k
    return folds, n_splits


def split_indices(obs, folds, n_splits, test_fold, donor_col='donor_id'):
    """
    Cell indices for train / val / test.
    test = `test_fold`, val = the next fold, train = all remaining folds.
    """
    val_fold = (test_fold + 1) % n_splits
    cell_fold = obs[donor_col].map(folds).to_numpy()
    test_idx = np.where(cell_fold == test_fold)[0]
    val_idx = np.where(cell_fold == val_fold)[0]
    train_idx = np.where((cell_fold != test_fold) & (cell_fold != val_fold))[0]
    return train_idx, val_idx, test_idx


def describe_split(obs, idx_dict, donor_col='donor_id', label_col='tangle_label'):
    """Table of donors and cells per stage for each split (for sanity checks)."""
    rows = []
    for name, idx in idx_dict.items():
        sub = obs.iloc[idx]
        for label, g in sub.groupby(label_col, observed=True):
            rows.append({'split': name, 'label': label,
                         'n_donors': g[donor_col].nunique(), 'n_cells': len(g)})
    return pd.DataFrame(rows).pivot(index='label', columns='split',
                                    values=['n_donors', 'n_cells']).fillna(0).astype(int)
