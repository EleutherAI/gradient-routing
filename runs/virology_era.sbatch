#!/bin/bash
#SBATCH --job-name=virology-era
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/a6a/lucia.a6a/gradient-routing/runs/virology-era-%j.out

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate gradient-routing

module load PrgEnv-cray
module load cuda/12.6

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/home/a6a/lucia.a6a/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/home/a6a/lucia.a6a/.cache/torch_inductor
export HF_HUB_OFFLINE=1

cd /home/a6a/lucia.a6a/gradient-routing

echo "=== Virology ERA Gradient Routing ==="
echo "Time: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# Step 1: ERA training (single GPU, picked by get_gpu_with_most_memory)
echo "Step 1: ERA training"
python projects/virology_era/virology_era.py

SAVE_DIR="projects/virology_era/outputs/full_run"

if [ ! -f "${SAVE_DIR}/final.pt" ]; then
    echo "ERROR: Final model not found at ${SAVE_DIR}/final.pt"
    exit 1
fi

# Step 2: Convert TL model to HF format for evaluation
echo "Step 2: Convert to HF format"
python projects/virology_era/evaluate_virology.py \
    "${SAVE_DIR}/final.pt" \
    --hf-convert \
    --output-dir "${SAVE_DIR}/hf_model"

HF_MODEL="${SAVE_DIR}/hf_model"

if [ ! -d "$HF_MODEL" ]; then
    echo "ERROR: HF model not found at $HF_MODEL"
    exit 1
fi

# Step 3: Multi-GPU evaluation with lm_eval
echo "Step 3: WMDP Bio Robust evaluation"
python -m lm_eval --model hf \
    --model_args pretrained=$HF_MODEL \
    --tasks wmdp_bio_robust \
    --include_path "/home/a6a/lucia.a6a/unlearn/unlearn/lm_eval_tasks" \
    --batch_size 32 \
    --verbosity WARNING \
    --output_path "${SAVE_DIR}/eval_wmdp"

echo "Step 4: MMLU evaluation"
python -m lm_eval --model hf \
    --model_args pretrained=$HF_MODEL \
    --tasks mmlu \
    --num_fewshot 1 \
    --batch_size 32 \
    --verbosity WARNING \
    --output_path "${SAVE_DIR}/eval_mmlu"

# Also eval the baseline for comparison
echo "Step 5: Baseline WMDP Bio Robust"
BASELINE="/home/a6a/lucia.a6a/.cache/huggingface/hub/models--EleutherAI--deep-ignorance-unfiltered/snapshots/c8df368ff247cb90b62e21e1689260701b3ff25a"
python -m lm_eval --model hf \
    --model_args pretrained=$BASELINE \
    --tasks wmdp_bio_robust \
    --include_path "/home/a6a/lucia.a6a/unlearn/unlearn/lm_eval_tasks" \
    --batch_size 32 \
    --verbosity WARNING \
    --output_path "${SAVE_DIR}/eval_wmdp_baseline"

echo "Step 6: Baseline MMLU"
python -m lm_eval --model hf \
    --model_args pretrained=$BASELINE \
    --tasks mmlu \
    --num_fewshot 1 \
    --batch_size 32 \
    --verbosity WARNING \
    --output_path "${SAVE_DIR}/eval_mmlu_baseline"

echo "Done: $(date)"
