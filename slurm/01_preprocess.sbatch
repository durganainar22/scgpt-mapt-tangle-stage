#!/bin/bash
#SBATCH --job-name=scgpt_prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Step 1 (CPU): preprocessing notebook + non-deep-learning baselines.
# Submit from the project directory:   sbatch slurm/01_preprocess.sbatch

set -euo pipefail
export SCGPT_PROJECT_DIR="${SCGPT_PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$SCGPT_PROJECT_DIR"
mkdir -p logs executed

# --- environment (edit if your setup differs) ---
module load anaconda3/2022.05 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
unset PYTHONPATH
conda activate scgpt_mapt

jupyter nbconvert --to notebook --execute scgpt_preprocessing.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir executed --output scgpt_preprocessing_executed.ipynb

python baselines.py
