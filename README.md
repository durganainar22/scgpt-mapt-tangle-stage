# scGPT Fine-tuning for MAPT Tangle Stage Prediction in Alzheimer's Disease

## Overview

This project fine-tunes scGPT, a transformer-based single-cell foundation model pretrained on 33 million human cells, to predict neurofibrillary tangle stage (Braak NFT stage) from single-nucleus RNA-seq of human Alzheimer's disease brain tissue.

Neurofibrillary tangles of hyperphosphorylated MAPT (tau) are a defining pathology of Alzheimer's disease. The question: **can a foundation model recognise a donor's tangle stage from the transcriptional state of individual neurons, in donors it has never seen?**

> **Key result (v2):** evaluated on **held-out donors**, neither fine-tuned scGPT nor any baseline predicts tangle stage better than chance (fine-tuned scGPT balanced accuracy 0.249 ± 0.056, macro AUC 0.502 ± 0.092; chance = 0.25 / 0.50). v1's reported 48% accuracy / AUC 0.68 came from a cell-level split (cells of the same donor in train and test) plus broken model inputs, and is withdrawn. See [Results](#results) and [Changes from v1](#changes-from-v1).

---

## Repository Structure

    scgpt-mapt-tangle-stage/
    ├── scgpt_preprocessing.ipynb   # Data loading, donor-level EDA, QC, MAPT pseudobulk, baseline UMAP
    ├── scgpt_finetune.ipynb        # Donor-level split, scGPT tokenization, frozen probe, fine-tuning, evaluation
    ├── splits.py                   # Donor-level cross-validation folds (shared by all models)
    ├── metrics_utils.py            # Imbalance-aware metrics (balanced acc, macro-F1, donor-level acc)
    ├── baselines.py                # Majority-class and PCA + logistic regression baselines (CPU)
    ├── aggregate_results.py        # Mean ± SD across folds -> results_summary.json
    ├── results_summary.json        # All metrics, summary + per run
    ├── slurm/
    │   ├── 01_preprocess.sbatch    # CPU: preprocessing notebook + baselines
    │   └── 02_finetune_array.sbatch# GPU: one fine-tuning run per donor fold
    ├── results/                    # per-fold JSON results + summary.md
    ├── executed/                   # executed notebooks (outputs), one per run
    ├── figures/                    # MAPT pseudobulk, baseline UMAP, per-run confusion matrices + test-donor UMAPs
    └── README.md

Not included due to size: raw GEO files, `neurons_hvg_preprocessed.h5ad`, `scgpt_human_model/`, checkpoints.

---

## Dataset

Morabito et al. 2021, *Nature Genetics* 53:1143–1155 — GEO **GSE174367**

- 61,472 nuclei from post-mortem human prefrontal cortex of 18 donors; 12,331 excitatory + inhibitory neurons used here (39–1,641 per donor)
- "Tangle stage" is the donor's **Braak neurofibrillary tangle (NFT) stage** (Braak & Braak 1991). Stages present: 1, 2, 5, 6 (stages 3–4 absent)
- **Tangle stage is a donor-level label.** The number of independent samples is the number of donors (18), not the number of cells.
- **Stage is fully confounded with diagnosis:** every Stage 1–2 donor is a Control and every Stage 5–6 donor has AD. Sequencing batch is spread across stages (each stage has donors from 2–3 of the 3 batches).

| Stage | Donors | Diagnosis | Donors per batch (1 / 2 / 3) |
|---|---|---|---|
| 1 | 3 | Control | 1 / 1 / 1 |
| 2 | 4 | Control | 1 / 1 / 2 |
| 5 | 3 | AD | 0 / 1 / 2 |
| 6 | 8 | AD | 2 / 3 / 3 |

---

## Methods

### Preprocessing (`scgpt_preprocessing.ipynb`)
- Keep donor ID from the metadata (needed for donor-level splits)
- Subset to EX + INH neurons (tangles form primarily in neurons)
- QC: ≥200 genes per cell, ≥10 cells per gene; raw counts kept in `layers['counts']`
- Normalize to 10,000 counts + log1p (kept in `X`, **not scaled**)
- 3,000 highly variable genes + MAPT force-included
- MAPT across stages: log-normalized per cell (visualization) and **pseudobulk CPM per donor**, tested with donors as the unit (Kruskal–Wallis, Spearman)
- PCA/UMAP computed on a scaled **copy**

### Evaluation design
- **Donor-level cross-validation** (`splits.py`): donors are assigned to folds stratified by stage; `n_splits = min(5, fewest donors in any stage)`. For each fold: test = that fold, validation = next fold, training = the rest. No donor appears in more than one split (asserted). With 18 donors this gives 3 folds, so each run trains on **6 donors**, selects the checkpoint on 6 and tests on 6.
- **Metrics** (`metrics_utils.py`): balanced accuracy and macro-F1 as headline numbers (Stage 6 is ~52% of cells, so plain accuracy rewards always predicting Stage 6), per-class recall, one-vs-rest AUC, and **donor-level accuracy** (mean predicted probabilities per test donor).
- **Baselines on identical folds:** majority class; PCA (50) + logistic regression; linear probe on **frozen** pretrained scGPT embeddings.

### scGPT fine-tuning (`scgpt_finetune.ipynb`)
- Tokenization with scGPT's own tools: `Preprocessor(binning=51)` (per-cell quantile binning, as in pretraining) and `tokenize_and_pad_batch(append_cls=True)` — `<cls>` token at position 0, zero-expression genes dropped, padding with `<pad>` / `pad_value=-2`, up to `MAX_LEN=513` tokens per cell
- Pretrained whole-human checkpoint; attention weights renamed from flash-attention (`Wqkv`) to PyTorch (`in_proj`) naming so they load; loading asserts that only the new classifier head is randomly initialized
- Trainable: last 2 transformer layers + classification head
- Class imbalance: `WeightedRandomSampler` only; plain cross-entropy
- AdamW (lr 1e-4, weight decay 0.01), gradient clipping 1.0, LR halved on plateau, early stopping (patience 3) and checkpoint selection on **validation macro-F1**

---

## Results

18 donors (Stage 1: 3, Stage 2: 4, Stage 5: 3, Stage 6: 8) → 3 donor-level folds. scGPT: 3 folds × 2 seeds; baselines: 3 folds. Mean ± SD across runs (`results/summary.md`, `results_summary.json`).

| Model | Balanced accuracy | Macro-F1 | Macro AUC | Accuracy | Donor-level accuracy |
|---|---|---|---|---|---|
| Majority class | 0.250 ± 0.000 | 0.173 ± 0.044 | 0.500 ± 0.000 | 0.545 ± 0.201 | 0.444 ± 0.096 |
| PCA (50) + logistic regression | 0.275 ± 0.083 | 0.237 ± 0.043 | 0.514 ± 0.070 | 0.316 ± 0.113 | 0.222 ± 0.255 |
| Frozen scGPT embedding + LR | 0.237 ± 0.026 | 0.201 ± 0.020 | 0.488 ± 0.028 | 0.276 ± 0.060 | 0.194 ± 0.125 |
| **Fine-tuned scGPT** | **0.249 ± 0.056** | **0.197 ± 0.031** | **0.502 ± 0.092** | 0.301 ± 0.051 | 0.389 ± 0.136 |

Chance: balanced accuracy 0.25, AUC 0.50. Accuracy is inflated by the Stage 6 majority and is shown for reference only.

**AD vs Control (post-hoc).** Because stage and diagnosis are confounded, the 4-class predictions were also collapsed to Control (Stages 1–2) vs AD (Stages 5–6), from the saved confusion matrices and donor-level predictions. No model was retrained for this.

| Model | Balanced accuracy (cells) | Donor-level accuracy |
|---|---|---|
| Majority class | 0.500 ± 0.000 | 0.611 ± 0.096 |
| PCA (50) + logistic regression | 0.495 ± 0.042 | 0.500 ± 0.289 |
| Frozen scGPT embedding + LR | 0.440 ± 0.038 | 0.333 ± 0.149 |
| Fine-tuned scGPT | 0.469 ± 0.054 | 0.583 ± 0.175 |

**MAPT expression.** Donor pseudobulk MAPT (CPM) does not change with tangle stage (Kruskal–Wallis H = 1.44, p = 0.70; Spearman ρ = 0.03, p = 0.91; n = 18 donors; `figures/MAPT_by_tangle_stage.png`, `figures/MAPT_pseudobulk_by_donor.csv`). This is expected: tangles are hyperphosphorylated, aggregated tau protein, not higher MAPT mRNA.

Per-run confusion matrices and test-donor UMAPs: `figures/confusion_matrix_fold*_seed*.png`, `figures/UMAP_test_donors_fold*_seed*.png`.

**Interpretation**
- In unseen donors, tangle stage is **not predictable** from neuronal transcriptomes in this dataset — not by fine-tuned scGPT, pretrained scGPT embeddings, or a linear model on PCA features. Fine-tuned scGPT is not better than always predicting the most common stage, even at the donor level. Even the coarser AD-vs-Control split (which stage fully determines here) is at chance.
- Fine-tuning fits training donors quickly while loss on new (validation) donors stays much higher (e.g. fold 0, epoch 1: train 0.90 vs validation 3.31) — the expected signature of learning donor-specific features rather than a transferable pathology signal. Per-epoch curves are in the executed notebooks (`executed/`).
- The v1 result was therefore an artefact of donor leakage (and inputs that did not match scGPT's pretraining). This is the main lesson of the project: with donor-level labels, the unit of generalisation is the donor, and the effective sample size here is 18.
- A negative result at n = 18 donors (6 per training set) does not show that no stage signal exists — only that it is not detectable/transferable at this sample size, with neurons only and 3 of 4 stages represented by 3–4 donors. Larger cohorts (e.g. SEA-AD, ROSMAP snRNA-seq), pseudobulk donor-level models, and ordinal or continuous pathology targets are the natural next steps.

---

## Changes from v1

| # | v1 problem | Why it mattered | v2 fix |
|---|---|---|---|
| 1 | Random **cell-level** split | Cells of the same donor in train and test → model can recognise donors instead of pathology; inflated scores | Donor-level folds, stratified by stage; leakage asserted absent |
| 2 | Pretrained attention weights **not loaded** (`Wqkv` vs `in_proj` key names, hidden by `strict=False`; the "34 missing keys") | All 12 attention layers were random; "fine-tuning a foundation model" was not really happening | Keys renamed; assertion that only `cls_decoder` is new |
| 3 | `sc.pp.scale` ran **in place** before saving | scGPT received z-scores; "top expressed genes" meant "above gene mean" | Scaling applied to a copy; `X` stays log1p (asserted ≥ 0) |
| 4 | No `<cls>` token | `cell_emb_style='cls'` read position 0 = an arbitrary gene | `tokenize_and_pad_batch(append_cls=True)` |
| 5 | Padding id 0, mask on `vocab['<pad>']`; `PAD_VALUE` unused | Padding attended to as real genes | Pad with `<pad>` id and `pad_value` |
| 6 | Per-gene binning across cells | Did not match per-cell binning used in pretraining | scGPT `Preprocessor(binning=51)` |
| 7 | Sampler + class-weighted loss + manual 3× Stage-6 weight | Double correction, then a patch tuned on results | Sampler only, plain CE, model selection on validation donors |
| 8 | Compared to 25% random baseline | Majority class already gives ~52% accuracy | Majority, PCA+LR and frozen-probe baselines; balanced metrics |
| 9 | MAPT boxplot of raw counts, cells as samples | Confounded by sequencing depth; pseudo-replication | Normalized + donor pseudobulk, donor-level tests |
| 10 | Correctness UMAP over all cells (mostly training) | Showed memorisation, not generalisation | Test-donor cells only, also coloured by donor |

v1 biological interpretations ("Stage 5 is transitional", "Stage 1/6 confusion reflects survivor bias", "MAPT peak reflects survivor bias") are withdrawn as findings; they can be revisited as hypotheses once v2 results are in.

---

## Environment Setup (Northeastern Explorer HPC)

    srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 --pty bash
    module load anaconda3/2024.06
    conda create -n scgpt_mapt python=3.10 -y
    conda activate scgpt_mapt
    pip install "scgpt==0.2.2" "scanpy==1.9.8" "anndata==0.10.8" "torch==2.2.2" "torchtext==0.17.2" \
        "huggingface_hub==0.23.4" "datasets==2.20.0" "matplotlib==3.7.5" "seaborn==0.13.2" \
        umap-learn ipykernel nbconvert scikit-learn
    unset PYTHONPATH
    python -m ipykernel install --user --name scgpt_mapt --display-name "scgpt_mapt"

Pretrained model: whole-human checkpoint from the [scGPT Model Zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo) → `scgpt_human_model/{best_model.pt, args.json, vocab.json}`

Data:

    wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174367/suppl/GSE174367_snRNA-seq_filtered_feature_bc_matrix.h5"
    wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174367/suppl/GSE174367_snRNA-seq_cell_meta.csv.gz"

---

## How To Run

From the project directory (set `SCGPT_PROJECT_DIR` if data lives elsewhere):

    # 1. CPU: preprocessing + baselines  (check the log for donors per stage and n_splits)
    sbatch slurm/01_preprocess.sbatch

    # 2. GPU: test one fold first, then the rest (18 donors -> 3 folds: 0-2), then a second seed
    sbatch --array=0 slurm/02_finetune_array.sbatch
    sbatch --array=1-2 slurm/02_finetune_array.sbatch
    SEED=7 sbatch --array=0-2 slurm/02_finetune_array.sbatch

    # 3. Combine
    python aggregate_results.py

Executed notebooks (with outputs) are written to `executed/`.

---

## Limitations

- 18 donors (6 per training set): per-stage conclusions rest on very few individuals; fold-to-fold variance is large
- Stages 3–4 absent; stage is fully confounded with diagnosis (Stages 1–2 Control, 5–6 AD), so stage and AD effects cannot be separated
- Very uneven neurons per donor (39–1,641); cell-level metrics weight donors unequally (donor-level accuracy is reported for this reason)
- Neurons only — glial responses to tau are excluded
- Post-mortem tissue; no correction for PMI / RIN (PMI is missing for one donor)
- Single dataset; no external cohort

---

## References

- Braak & Braak (1991) Neuropathological stageing of Alzheimer-related changes. *Acta Neuropathologica* 82:239–259

- Cui et al. (2024) scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*
- Morabito et al. (2021) Single-nucleus chromatin accessibility and transcriptomic characterization of Alzheimer's disease. *Nature Genetics*

## Author

Durga Gomathi Arumuganainar — MS Bioinformatics, Northeastern University — github.com/durganainar22
