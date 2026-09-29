#!/usr/bin/env python
"""
v3 Step 5c (GPU): fine-tune scGPT for Braak V-VI vs 0-IV, one development fold per run (FOLD = 0..4).
The locked test donors are dropped on load and never touched.

Per run:
  test donors  = development fold FOLD
  val donors   = 20% of the other development donors (stratified by label, fixed seed) - early stopping only
  train donors = the rest (~43 donors, ~21k cells)
Donor-level prediction = mean of the cell probabilities; early stopping on donor-level validation AUC
(the unit the models are evaluated on).

Tokenisation and weight loading are identical to 05a (HVGs chosen on development donors, scGPT Preprocessor
binning, <cls> token, Wqkv -> in_proj renaming with an assertion that only the classifier head is new).
Training as in v2: last UNFREEZE_LAST_N transformer layers + classifier head trainable, AdamW, grad clip 1.0,
WeightedRandomSampler balancing the two labels, plain cross-entropy.

Inputs:  data/seaad_neurons_sub500.h5ad, seaad/results/02_splits.csv, scgpt_human_model/
Outputs: seaad/results/05c_fold{FOLD}_seed{SEED}.json   (config, split, history, donor predictions)
         checkpoints/seaad_ft_fold{FOLD}_seed{SEED}.pt  (best model, gitignored)

Usage:   FOLD=0 python seaad/05c_finetune.py   (SEED, MAX_LEN, N_EPOCHS, PATIENCE, BATCH_SIZE, LR, UNFREEZE_LAST_N)
"""
import json
import os
import random
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings('ignore')
PROJECT_DIR = os.environ.get('SCGPT_PROJECT_DIR',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
MODEL_DIR = os.environ.get('MODEL_DIR', 'scgpt_human_model')
FOLD = int(os.environ.get('FOLD', 0))
SEED = int(os.environ.get('SEED', 42))
MAX_LEN = int(os.environ.get('MAX_LEN', 513))
N_EPOCHS = int(os.environ.get('N_EPOCHS', 10))
PATIENCE = int(os.environ.get('PATIENCE', 3))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
LR = float(os.environ.get('LR', 1e-4))
UNFREEZE_LAST_N = int(os.environ.get('UNFREEZE_LAST_N', 2))
N_HVG, SPLIT_SEED = 3000, 42
PAD_TOKEN, CLS_TOKEN = '<pad>', '<cls>'
RUN = f'fold{FOLD}_seed{SEED}'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('checkpoints', exist_ok=True)
print(f'Run {RUN} | device {device} | GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

# ---------------------------------------------------------------- data + split (development donors only)
adata = ad.read_h5ad('data/seaad_neurons_sub500.h5ad')
adata = adata[adata.obs['split'] != 'test'].copy()
assert not (adata.obs['split'] == 'test').any()
adata.X = adata.X.astype(np.float32)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
hvg = sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat', inplace=False)
genes = adata.var_names[hvg['highly_variable'].to_numpy()]    # dev donors only, as in 05a

donors = adata.obs.groupby('Donor ID', observed=True).agg(label=('label', 'first'), split=('split', 'first'))
test_d = donors.index[donors['split'] == f'fold{FOLD}']
rest = donors.index[donors['split'] != f'fold{FOLD}']
train_d, val_d = train_test_split(rest, test_size=0.2, stratify=donors.loc[rest, 'label'], random_state=SPLIT_SEED)
assert not (set(train_d) & set(val_d) or set(train_d) & set(test_d) or set(val_d) & set(test_d))
donor_of = adata.obs['Donor ID'].astype(str).to_numpy()
labels = adata.obs['label'].astype(int).to_numpy()
idx = {k: np.where(np.isin(donor_of, list(v)))[0] for k, v in
       [('train', train_d), ('val', val_d), ('test', test_d)]}
for k, v in [('train', train_d), ('val', val_d), ('test', test_d)]:
    print(f'{k:5s}: {len(v)} donors ({int(donors.loc[v, "label"].sum())} V-VI), {len(idx[k]):,} cells')

# ---------------------------------------------------------------- tokenise (identical to 05a)
with open(f'{MODEL_DIR}/args.json') as f:
    model_args = json.load(f)
with open(f'{MODEL_DIR}/vocab.json') as f:
    vocab = json.load(f)
PAD_VALUE = model_args.get('pad_value', -2)
genes = [g for g in genes if g in vocab]
adata = adata[:, genes].copy()
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch

if issparse(adata.X):
    adata.X = adata.X.toarray().astype(np.float32)
Preprocessor(use_key='X', filter_gene_by_counts=False, filter_cell_by_counts=False, normalize_total=False,
             log1p=False, subset_hvg=False, binning=model_args['n_bins'],
             result_binned_key='X_binned')(adata, batch_key=None)
tok = tokenize_and_pad_batch(np.asarray(adata.layers['X_binned']),
                             np.array([vocab[g] for g in adata.var_names], dtype=int),
                             max_len=MAX_LEN, vocab=vocab, pad_token=PAD_TOKEN, pad_value=PAD_VALUE,
                             append_cls=True, include_zero_gene=False, cls_token=CLS_TOKEN, return_pt=True)
all_genes, all_values = tok['genes'], tok['values']
assert (all_genes[:, 0] == vocab[CLS_TOKEN]).all()
print(f'Tokenised {tuple(all_genes.shape)} with {len(genes)} HVGs in vocab')


class Cells(Dataset):
    def __init__(self, ix):
        self.ix = np.asarray(ix)

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        j = self.ix[i]
        return {'gene_ids': all_genes[j], 'values': all_values[j], 'label': int(labels[j]), 'cell': int(j)}


tr_lab = labels[idx['train']]
w = (1.0 / np.bincount(tr_lab, minlength=2))[tr_lab]
sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr_lab),
                                replacement=True, generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(Cells(idx['train']), batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(Cells(idx['val']), batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(Cells(idx['test']), batch_size=64, shuffle=False, num_workers=2)

# ---------------------------------------------------------------- model (identical loading to 05a / v2)
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
model = model.to(device)

for p in model.parameters():
    p.requires_grad = False
unfreeze = ['cls_decoder'] + [f'transformer_encoder.layers.{model_args["nlayers"] - 1 - i}.'
                              for i in range(UNFREEZE_LAST_N)]
for n, p in model.named_parameters():
    if any(u in n for u in unfreeze):
        p.requires_grad = True
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {n_train:,} / {sum(p.numel() for p in model.parameters()):,} ({unfreeze})')


def forward(batch):
    g = batch['gene_ids'].to(device)
    v = batch['values'].to(device).float()
    return model(g, v, src_key_padding_mask=g.eq(vocab[PAD_TOKEN]), batch_labels=None,
                 CLS=True, CCE=False, MVC=False, ECS=False, do_sample=False)['cls_output']


@torch.no_grad()
def predict(loader):
    model.eval()
    probs, cells = [], []
    for b in loader:
        probs.append(torch.softmax(forward(b).float(), 1)[:, 1].cpu().numpy())
        cells.append(b['cell'].numpy())
    probs, cells = np.concatenate(probs), np.concatenate(cells)
    donor_p = pd.Series(probs).groupby(donor_of[cells]).mean()
    donor_y = pd.Series(labels[cells]).groupby(donor_of[cells]).first()
    return probs, cells, donor_p, donor_y


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
ckpt = f'checkpoints/seaad_ft_{RUN}.pt'
hist = {'train_loss': [], 'val_cell_loss': [], 'val_donor_auc': [], 'val_cell_auc': []}
best, bad = -1.0, 0

for epoch in range(N_EPOCHS):
    t0 = time.time()
    model.train()
    run_loss = 0.0
    for b in train_loader:
        optimizer.zero_grad()
        loss = criterion(forward(b), b['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run_loss += loss.item()
    vp, vc, vdp, vdy = predict(val_loader)
    vy = labels[vc]
    hist['train_loss'].append(run_loss / len(train_loader))
    hist['val_cell_loss'].append(float(-np.mean(vy * np.log(vp + 1e-9) + (1 - vy) * np.log(1 - vp + 1e-9))))
    hist['val_donor_auc'].append(float(roc_auc_score(vdy, vdp)))
    hist['val_cell_auc'].append(float(roc_auc_score(vy, vp)))
    scheduler.step(hist['val_donor_auc'][-1])
    print(f'Epoch {epoch + 1:2d} | train loss {hist["train_loss"][-1]:.4f} | val cell loss '
          f'{hist["val_cell_loss"][-1]:.4f} | val cell AUC {hist["val_cell_auc"][-1]:.3f} | '
          f'val donor AUC {hist["val_donor_auc"][-1]:.3f} | {time.time() - t0:.0f}s', flush=True)
    if hist['val_donor_auc'][-1] > best:
        best, bad = hist['val_donor_auc'][-1], 0
        torch.save(model.state_dict(), ckpt)
        print(f'  -> best so far (val donor AUC {best:.3f}), saved', flush=True)
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f'Early stopping after epoch {epoch + 1}')
            break

model.load_state_dict(torch.load(ckpt, map_location=device))
tp, tc, tdp, tdy = predict(test_loader)
test_donor_auc = roc_auc_score(tdy, tdp) if tdy.nunique() == 2 else float('nan')
print(f'Test fold {FOLD}: donor AUC {test_donor_auc:.3f} over {len(tdp)} donors; cell AUC '
      f'{roc_auc_score(labels[tc], tp):.3f}  (single fold - the pooled result is computed in 05d)')

out = {'run': RUN, 'fold': FOLD, 'seed': SEED,
       'config': {'max_len': MAX_LEN, 'n_epochs_max': N_EPOCHS, 'epochs_trained': len(hist['train_loss']),
                  'best_epoch': int(np.argmax(hist['val_donor_auc']) + 1), 'patience': PATIENCE,
                  'batch_size': BATCH_SIZE, 'lr': LR, 'unfrozen_layers': UNFREEZE_LAST_N, 'n_hvg_in_vocab': len(genes),
                  'early_stopping': 'val donor AUC (mean cell probability per donor)',
                  'imbalance': 'WeightedRandomSampler over labels'},
       'split': {'train_donors': sorted(train_d), 'val_donors': sorted(val_d), 'test_donors': sorted(test_d),
                 'n_train_cells': int(len(idx['train'])), 'n_val_cells': int(len(idx['val'])),
                 'n_test_cells': int(len(idx['test']))},
       'history': hist,
       'test_donor_prob': {d: float(p) for d, p in tdp.items()},
       'test_donor_auc_this_fold': None if np.isnan(test_donor_auc) else float(test_donor_auc)}
with open(f'seaad/results/05c_{RUN}.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'Saved seaad/results/05c_{RUN}.json')
