#!/usr/bin/env python
"""
v3 Step 3: pull neurons out of the 84 SEA-AD per-donor files, one donor at a time
(the full atlas is ~33 GB, so nothing is ever loaded all at once).

Per donor, from neurons with `Used in analysis` == True (Class = Neuronal: Glutamatergic / GABAergic):
  - pseudobulk: summed raw UMIs over ALL neurons (donor x gene) and per neuron subclass
  - a random subsample of N_PER_DONOR neurons (raw UMIs) for scGPT and cell-level baselines;
    donors with fewer neurons contribute all of them

Raw counts come from layers['UMIs'] (X in these files is ln(UP10K+1)). All genes are kept;
gene selection happens later, on development donors only.

Input:   data/donors/*.h5ad   (bash seaad/00_download.sh donors)
         seaad/results/02_splits.csv
Outputs: data/seaad_neurons_sub{N}.h5ad        cells x genes, X = raw UMIs (sparse), obs = donor + labels
         data/seaad_pseudobulk_donor.h5ad      donors x genes, raw UMI sums
         data/seaad_pseudobulk_subclass.h5ad   (donor, subclass) x genes, raw UMI sums
         seaad/results/03_extract.txt

Usage:   python seaad/03_extract_neurons.py            (N_PER_DONOR env var, default 500)
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
N_PER_DONOR = int(os.environ.get('N_PER_DONOR', 500))
SEED = 42
OBS_KEEP = ['Donor ID', 'Class', 'Subclass', 'Supertype', 'Number of UMIs', 'Genes detected',
            'Fraction mitochondrial UMIs', 'method', 'library_prep']

splits = pd.read_csv('seaad/results/02_splits.csv', index_col='Donor ID')
files = sorted(glob.glob('data/donors/*_SEAAD_MTG_RNAseq_final-nuclei.*.h5ad'))
file_donor = {os.path.basename(f).split('_SEAAD_')[0]: f for f in files}
missing = sorted(set(splits.index) - set(file_donor))
assert not missing, f'{len(missing)} donor files missing (download incomplete?): {missing[:5]}'

rng = np.random.default_rng(SEED)
genes = None
sub_parts, pb_donor, pb_sub, pb_sub_index, log = [], [], [], [], []

for i, donor in enumerate(splits.index):
    a = ad.read_h5ad(file_donor[donor])
    assert (a.obs['Donor ID'].astype(str) == donor).all(), donor
    if genes is None:
        genes = a.var_names.copy()
    assert a.var_names.equals(genes), f'{donor}: gene order differs'

    keep = (a.obs['Class'].astype(str).str.startswith('Neuronal')
            & a.obs['Used in analysis'].astype(bool)).to_numpy()
    counts = sp.csr_matrix(a.layers['UMIs'][keep], dtype=np.float32)
    obs = a.obs.loc[keep, OBS_KEEP].copy()
    del a

    pb_donor.append(np.asarray(counts.sum(axis=0)).ravel())
    for sc, idx in obs.groupby('Subclass', observed=True).indices.items():
        pb_sub.append(np.asarray(counts[idx].sum(axis=0)).ravel())
        pb_sub_index.append((donor, sc, len(idx)))

    n = counts.shape[0]
    pick = np.sort(rng.choice(n, size=min(N_PER_DONOR, n), replace=False))
    sub_parts.append(ad.AnnData(X=counts[pick], obs=obs.iloc[pick]))
    log.append((donor, n, len(pick)))
    print(f'[{i + 1}/{len(splits)}] {donor}: {n} neurons, kept {len(pick)}', flush=True)

var = pd.DataFrame(index=genes)
lab_cols = ['split', 'label', 'braak_num', 'Braak', 'dementia']

cells = ad.concat(sub_parts, index_unique=None)
cells.var = var
cells.obs = cells.obs.join(splits[lab_cols], on='Donor ID')
cells.obs_names_make_unique()
cells.write_h5ad(f'data/seaad_neurons_sub{N_PER_DONOR}.h5ad', compression='gzip')

pbd = ad.AnnData(X=np.vstack(pb_donor).astype(np.float32),
                 obs=splits.loc[list(splits.index), lab_cols + ['n_neurons_used']], var=var)
pbd.write_h5ad('data/seaad_pseudobulk_donor.h5ad', compression='gzip')

sub_obs = pd.DataFrame(pb_sub_index, columns=['Donor ID', 'Subclass', 'n_cells'])
sub_obs = sub_obs.join(splits[lab_cols], on='Donor ID')
sub_obs.index = sub_obs['Donor ID'] + '|' + sub_obs['Subclass']
pbs = ad.AnnData(X=np.vstack(pb_sub).astype(np.float32), obs=sub_obs, var=var)
pbs.write_h5ad('data/seaad_pseudobulk_subclass.h5ad', compression='gzip')

with open('seaad/results/03_extract.txt', 'w', encoding='utf-8') as f:
    lg = pd.DataFrame(log, columns=['donor', 'n_neurons', 'n_kept']).set_index('donor')
    lines = [f'Donors: {len(lg)}  genes: {len(genes)}  N_PER_DONOR: {N_PER_DONOR}  seed: {SEED}',
             f'Subsample: {cells.n_obs:,} cells  (donors with < {N_PER_DONOR} neurons: '
             f'{int((lg.n_neurons < N_PER_DONOR).sum())})',
             f'Pseudobulk: {pbd.n_obs} donors; {pbs.n_obs} donor x subclass rows',
             '\n=== Subsampled cells per split x label ===',
             pd.crosstab(cells.obs['split'], cells.obs['label']).to_string(),
             '\n=== Subsampled cells per subclass ===',
             cells.obs['Subclass'].astype(str).value_counts().to_string()]
    f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
