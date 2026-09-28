#!/usr/bin/env python
"""
v3 Step 5a (GPU): frozen pretrained scGPT cell embeddings for the 41,719 subsampled neurons.
No training, no labels - one forward pass. The CPU probe (05b_embed_probe.py) does the evaluation.

Pipeline (same tokenisation and weight loading as v2, which were verified there):
  raw UMIs -> normalise to 1e4 + log1p -> 3000 HVGs chosen on DEVELOPMENT donors only (label-free, but test
  cells are kept out of every choice) -> genes in the scGPT vocab -> scGPT Preprocessor(binning = n_bins,
  per cell) -> tokenize_and_pad_batch(append_cls=True) -> <cls> embedding from the pretrained whole-human model

Inputs:  data/seaad_neurons_sub500.h5ad (03), scgpt_human_model/{best_model.pt,args.json,vocab.json}
Outputs: data/seaad_scgpt_frozen_emb.npy  (cells x d_model, float32, row order = the h5ad)
         data/seaad_scgpt_frozen_obs.csv  (cell obs: donor, subclass, split, label)
         seaad/results/05a_embed.txt

Usage:   python seaad/05a_embed.py        (MODEL_DIR, MAX_LEN, BATCH_SIZE env vars)
"""
import json
import os
import time
import warnings

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse

warnings.filterwarnings('ignore')
PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
MODEL_DIR = os.environ.get('MODEL_DIR', 'scgpt_human_model')
MAX_LEN = int(os.environ.get('MAX_LEN', 513))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
N_HVG, SEED = 3000, 42
PAD_TOKEN, CLS_TOKEN = '<pad>', '<cls>'
np.random.seed(SEED)                       # tokenize_and_pad_batch subsamples genes when > MAX_LEN
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = []


def say(s):
    print(s, flush=True)
    log.append(str(s))


say(f'Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

# ---------------------------------------------------------------- data
adata = ad.read_h5ad('data/seaad_neurons_sub500.h5ad')
adata.X = adata.X.astype(np.float32)
assert np.allclose(adata.X[:200].data, np.round(adata.X[:200].data)), 'X should be raw UMI counts'
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
dev = (adata.obs['split'] != 'test').to_numpy()
hvg = sc.pp.highly_variable_genes(adata[dev], n_top_genes=N_HVG, flavor='seurat', inplace=False)
genes = adata.var_names[hvg['highly_variable'].to_numpy()]

with open(f'{MODEL_DIR}/args.json') as f:
    model_args = json.load(f)
with open(f'{MODEL_DIR}/vocab.json') as f:
    vocab = json.load(f)
PAD_VALUE = model_args.get('pad_value', -2)
genes = [g for g in genes if g in vocab]
adata = adata[:, genes].copy()
say(f'Cells: {adata.n_obs:,} | HVGs (dev donors) in scGPT vocab: {len(genes)} / {N_HVG} | MAPT: {"MAPT" in genes}')

# ---------------------------------------------------------------- tokenise (scGPT's own tools)
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch

if issparse(adata.X):                      # scgpt 0.2.2 uses sparse.A, removed in SciPy >= 1.14
    adata.X = adata.X.toarray().astype(np.float32)
Preprocessor(use_key='X', filter_gene_by_counts=False, filter_cell_by_counts=False, normalize_total=False,
             log1p=False, subset_hvg=False, binning=model_args['n_bins'],
             result_binned_key='X_binned')(adata, batch_key=None)
binned = np.asarray(adata.layers['X_binned'])
gene_ids = np.array([vocab[g] for g in adata.var_names], dtype=int)
n_expr = (binned > 0).sum(1)
say(f'Expressed HVGs per cell: median {int(np.median(n_expr))}; '
    f'{(n_expr > MAX_LEN - 1).mean() * 100:.1f}% of cells above MAX_LEN-1 = {MAX_LEN - 1} (genes subsampled)')
tok = tokenize_and_pad_batch(binned, gene_ids, max_len=MAX_LEN, vocab=vocab, pad_token=PAD_TOKEN,
                             pad_value=PAD_VALUE, append_cls=True, include_zero_gene=False,
                             cls_token=CLS_TOKEN, return_pt=True)
all_genes, all_values = tok['genes'], tok['values']
assert (all_genes[:, 0] == vocab[CLS_TOKEN]).all(), '<cls> must be at position 0'
del binned

# ---------------------------------------------------------------- model (v2 loading, verified)
model = TransformerModel(
    ntoken=len(vocab), d_model=model_args['embsize'], nhead=model_args['nheads'], d_hid=model_args['d_hid'],
    nlayers=model_args['nlayers'], nlayers_cls=3, n_cls=2, vocab=vocab, dropout=model_args['dropout'],
    pad_token=PAD_TOKEN, pad_value=PAD_VALUE, do_mvc=False, do_dab=False, use_batch_labels=False,
    domain_spec_batchnorm=False, input_emb_style='continuous', n_input_bins=model_args['n_bins'],
    cell_emb_style='cls', ecs_threshold=0.0, explicit_zero_prob=False, use_fast_transformer=False,
    pre_norm=False)
pretrained = torch.load(f'{MODEL_DIR}/best_model.pt', map_location='cpu')
pretrained = {k.replace('self_attn.Wqkv.', 'self_attn.in_proj_'): v for k, v in pretrained.items()}
state = model.state_dict()
loadable = {k: v for k, v in pretrained.items() if k in state and v.shape == state[k].shape}
state.update(loadable)
model.load_state_dict(state)
new = [k for k in state if k not in loadable]
assert all(k.startswith('cls_decoder') for k in new), f'Pretrained weights missing for: {new}'
say(f'Loaded {len(loadable)} pretrained tensors; only the unused classifier head is new ({len(new)} tensors)')
model = model.to(device).eval()

# ---------------------------------------------------------------- embed
t0, embs = time.time(), []
with torch.no_grad():                      # full precision, as in v2
    for i in range(0, len(all_genes), BATCH_SIZE):
        g = all_genes[i:i + BATCH_SIZE].to(device)
        v = all_values[i:i + BATCH_SIZE].to(device).float()
        out = model(g, v, src_key_padding_mask=g.eq(vocab[PAD_TOKEN]), batch_labels=None,
                    CLS=False, CCE=False, MVC=False, ECS=False, do_sample=False)
        embs.append(out['cell_emb'].float().cpu().numpy())
        if (i // BATCH_SIZE) % 100 == 0:
            print(f'  {i:,} / {len(all_genes):,} cells  ({time.time() - t0:.0f}s)', flush=True)
emb = np.concatenate(embs).astype(np.float32)
assert emb.shape[0] == adata.n_obs and np.isfinite(emb).all()
say(f'Embeddings: {emb.shape} in {time.time() - t0:.0f}s; mean norm {np.linalg.norm(emb, axis=1).mean():.2f}; '
    f'mean per-dimension SD across cells {emb.std(0).mean():.3f} (0 would mean every cell looks the same)')

np.save('data/seaad_scgpt_frozen_emb.npy', emb)
adata.obs[['Donor ID', 'Subclass', 'Supertype', 'split', 'label', 'braak_num', 'dementia']] \
    .to_csv('data/seaad_scgpt_frozen_obs.csv')
os.makedirs('seaad/results', exist_ok=True)
open('seaad/results/05a_embed.txt', 'w', encoding='utf-8').write('\n'.join(log) + '\n')
say('Saved data/seaad_scgpt_frozen_emb.npy, data/seaad_scgpt_frozen_obs.csv, seaad/results/05a_embed.txt')
