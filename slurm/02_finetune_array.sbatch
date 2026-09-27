#!/bin/bash
#SBATCH --job-name=scgpt_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Step 2 (GPU): one fine-tuning run per donor fold (array index = test fold).
#
# Test ONE fold first:        sbatch --array=0 slurm/02_finetune_array.sbatch
# Then the rest:              sbatch --array=1-2 slurm/02_finetune_array.sbatch
# Extra seeds (optional):     SEED=7 sbatch slurm/02_finetune_array.sbatch
#
# The preprocessing notebook prints how many donor folds are possible
# (n_splits = min(5, fewest donors in any stage)); GSE174367 gives 3 -> --array=0-2.
# Knobs: MAX_LEN (default 513), N_EPOCHS (15), BATCH_SIZE (32), LR (1e-4).

set -euo pipefail
export SCGPT_PROJECT_DIR="${SCGPT_PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$SCGPT_PROJECT_DIR"
mkdir -p logs executed

export FOLD="${SLURM_ARRAY_TASK_ID:-0}"
export SEED="${SEED:-42}"

# --- environment (edit if your setup differs) ---
module load anaconda3/2024.06 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
unset PYTHONPATH
conda activate scgpt_mapt

nvidia-smi
echo "Fold ${FOLD}, seed ${SEED}"

jupyter nbconvert --to notebook --execute scgpt_finetune.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir executed --output "scgpt_finetune_fold${FOLD}_seed${SEED}.ipynb"
