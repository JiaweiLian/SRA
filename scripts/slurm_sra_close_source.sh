#!/bin/bash
#SBATCH --job-name=sra_close_source
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=10-00:00:00
#SBATCH --output=slurm_logs/sra_close_source/%x_%j.log
#SBATCH --error=slurm_logs/sra_close_source/%x_%j.log

set -euo pipefail

LOG_DIR=${LOG_DIR:-slurm_logs/sra_close_source}
mkdir -p "${LOG_DIR}"

# Allow overrides when submitting, but fall back to the debug configuration values.
PYTHON_BIN=$(command -v python)
: "${PYTHON_BIN:?python executable not found in PATH}"
SAVE_DIR=${SAVE_DIR:-./results/SRA/gpt-4-1106-preview__surrogate__vicuna_7b_v1_5/test_cases}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gpt-4-1106-preview__surrogate__vicuna_7b_v1_5}
BEHAVIOR_START_IDX=${BEHAVIOR_START_IDX:-0}
BEHAVIOR_END_IDX=${BEHAVIOR_END_IDX:-150}
BEHAVIOR_IDS_SUBSET=${BEHAVIOR_IDS_SUBSET:-}

export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://jeniya.cn/v1}
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the environment before submitting the job.}"
export OPENAI_API_KEY

mkdir -p "${SAVE_DIR}"

cmd=("${PYTHON_BIN}" -u generate_test_cases.py \
    --method_name SRA \
    --experiment_name "${EXPERIMENT_NAME}" \
    --save_dir "${SAVE_DIR}")

if [[ -n "${BEHAVIOR_START_IDX}" ]]; then
    cmd+=(--behavior_start_idx "${BEHAVIOR_START_IDX}")
fi

if [[ -n "${BEHAVIOR_END_IDX}" ]]; then
    cmd+=(--behavior_end_idx "${BEHAVIOR_END_IDX}")
fi

if [[ -n "${BEHAVIOR_IDS_SUBSET}" ]]; then
    cmd+=(--behavior_ids_subset "${BEHAVIOR_IDS_SUBSET}")
fi

printf 'Launching command:\n%s\n' "${cmd[*]}"

"${cmd[@]}"

# Usage （export OPENAI_API_KEY=sk-xxx beforehand）:
# sbatch scripts/slurm_sra_close_source.sh

# Example with custom ranges and log directory:
#   sbatch scripts/slurm_sra_close_source.sh