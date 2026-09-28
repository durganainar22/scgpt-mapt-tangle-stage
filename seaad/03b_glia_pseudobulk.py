#!/usr/bin/env python
"""
v3 Step 3b: donor x subclass pseudobulk for NON-neuronal cells (astrocytes, microglia, oligodendrocytes,
OPCs, vascular), one donor file at a time. Complements 03_extract_neurons.py, which kept neurons only.

Input:   data/donors/*.h5ad, seaad/results/02_splits.csv
Output:  data/seaad_pseudobulk_glia.h5ad   (donor, subclass) x genes, raw UMI sums

Usage:   python seaad/03b_glia_pseudobulk.py
"""
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)

splits = pd.read_csv('seaad/results/02_splits.csv', index_col='Donor ID')
files = sorted(glob.glob('data/donors/*_SEAAD_MTG_RNAseq_final-nuclei.*.h5ad'))
file_donor = {os.path.basename(f).split('_SEAAD_')[0]: f for f in files}
missing = sorted(set(splits.index) - set(file_donor))
assert not missing, f'{len(missing)} donor files missing: {missing[:5]}'

genes, rows, mats = None, [], []
for i, donor in enumerate(splits.index):
    a = ad.read_h5ad(file_donor[donor])
    if genes is None:
        genes = a.var_names.copy()
    assert a.var_names.equals(genes), f'{donor}: gene order differs'
    keep = ((a.obs['Class'].astype(str) == 'Non-neuronal and Non-neural')
            & a.obs['Used in analysis'].astype(bool)).to_numpy()
    counts = sp.csr_matrix(a.layers['UMIs'][keep], dtype=np.float32)
    sub = a.obs.loc[keep, 'Subclass'].astype(str).to_numpy()
    del a
    for sc in np.unique(sub):
        idx = np.where(sub == sc)[0]
        mats.append(np.asarray(counts[idx].sum(axis=0)).ravel())
        rows.append((donor, sc, len(idx)))
    print(f'[{i + 1}/{len(splits)}] {donor}: {counts.shape[0]} non-neuronal cells', flush=True)

obs = pd.DataFrame(rows, columns=['Donor ID', 'Subclass', 'n_cells'])
obs = obs.join(splits[['split', 'label', 'braak_num', 'Braak', 'dementia']], on='Donor ID')
obs.index = obs['Donor ID'] + '|' + obs['Subclass']
out = ad.AnnData(X=np.vstack(mats).astype(np.float32), obs=obs, var=pd.DataFrame(index=genes))
out.write_h5ad('data/seaad_pseudobulk_glia.h5ad', compression='gzip')
print(f'\nSaved data/seaad_pseudobulk_glia.h5ad: {out.n_obs} donor x subclass rows')
print(obs.groupby('Subclass')['n_cells'].agg(['count', 'median', 'min']).to_string())
