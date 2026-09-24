#!/usr/bin/env bash
set -euo pipefail

# SparkArena 7-task RDT-1B: 8xA800, global bs=64, 80k steps, save every 10k (8 ckpts).
WORKSPACE="${RDT_WORKSPACE:-/personal/xiangpc/0812_Xpolicylab_bench/RDT-1B}"
ADAPTER="${WORKSPACE}/Xpolicylab/policy/RDT_1B"
RDT="${ADAPTER}/rdt"
WEIGHTS="${WORKSPACE}/pretrain_model"
DATA_DIR="${ADAPTER}/data/spark0_bench_7task_0908"
LANG_DIR="${ADAPTER}/lang_embeds_0908_7task"
STATS_PATH="${ADAPTER}/data/dataset_stat_0908_7task_rdt.json"
OUTPUT_DIR="${RDT_OUTPUT_DIR:-${ADAPTER}/checkpoints_0908_7task/rdt-1b-0908_7task-joint54-bs64-80k}"
LOG_DIR="${RDT_LOG_DIR:-${WORKSPACE}/logs}"
WANDB_ENV="${RDT_WANDB_ENV:-${WORKSPACE}/env_cfg/wandb.env}"

if [[ -f "${WANDB_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV}"
fi
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }

for model_dir in t5-v1_1-xxl siglip-so400m-patch14-384 rdt-1b; do
  [[ -d "${WEIGHTS}/${model_dir}" ]] || { echo "Missing ${model_dir}" >&2; exit 1; }
done
[[ -d "${DATA_DIR}" ]] || { echo "Missing HDF5 data: ${DATA_DIR}" >&2; exit 1; }
[[ -s "${LANG_DIR}/empty_lang_embed.pt" ]] || { echo "Missing empty_lang_embed.pt" >&2; exit 1; }
[[ -s "${STATS_PATH}" ]] || { echo "Missing dataset stats: ${STATS_PATH}" >&2; exit 1; }

episode_count="$(python3 - <<PY
from pathlib import Path
print(sum(1 for _ in Path("${DATA_DIR}").rglob("episode_*.hdf5")))
PY
)"
[[ "${episode_count}" == "700" ]] || { echo "Expected 700 hdf5, got ${episode_count}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export PATH="/personal/miniconda3/envs/rdt_1b/bin:${PATH}"
export PYTHONPATH="${WORKSPACE}/Xpolicylab:${RDT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RDT_HDF5_DIR="${DATA_DIR}"
export RDT_LANG_EMBED_DIR="${LANG_DIR}"
export RDT_DATASET_STAT_PATH="${STATS_PATH}"
export RDT_DATASET_NAME=robodojo_aloha_hdf5
export RDT_DROP_SHORT_EPISODES=0
export RDT_PROGRESS_PATH="${OUTPUT_DIR}/progress_0908_7task.json"
export WANDB_PROJECT="${WANDB_PROJECT:-xpolicylab-0908-SparkArena}"
export WANDB_NAME="${WANDB_NAME:-rdt-1b-0908_7task-joint54-bs64-80k}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-rdt-1b-0908-7task-0912}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_DIR="${LOG_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_EXTENSIONS_DIR="${WORKSPACE}/.cache/torch_extensions_7task"
export TRITON_CACHE_DIR="${WORKSPACE}/.cache/triton_7task"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "[RDT-1B][SparkArena7] episodes=${episode_count} tasks=7 action=joint54"
echo "[RDT-1B][SparkArena7] per_gpu_bs=8 global_bs=64 steps=80000 save=10000 keep=8"
echo "[RDT-1B][SparkArena7] data=${DATA_DIR}"
echo "[RDT-1B][SparkArena7] output=${OUTPUT_DIR}"
echo "[RDT-1B][SparkArena7] wandb=${WANDB_PROJECT}/${WANDB_NAME} id=${WANDB_RUN_ID}"

cd "${RDT}"
exec deepspeed --num_gpus=8 --master_port="${RDT_MASTER_PORT:-29639}" main.py \
  --deepspeed=./configs/zero2.json \
  --pretrained_model_name_or_path="${WEIGHTS}/rdt-1b" \
  --pretrained_text_encoder_name_or_path="${WEIGHTS}/t5-v1_1-xxl" \
  --pretrained_vision_encoder_name_or_path="${WEIGHTS}/siglip-so400m-patch14-384" \
  --output_dir="${OUTPUT_DIR}" \
  --seed=42 \
  --train_batch_size=8 \
  --sample_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=80000 \
  --checkpointing_period=10000 \
  --sample_period=0 \
  --checkpoints_total_limit=8 \
  --lr_scheduler=constant \
  --learning_rate=1e-4 \
  --mixed_precision=bf16 \
  --dataloader_num_workers=4 \
  --image_aug \
  --dataset_type=finetune \
  --state_noise_snr=40 \
  --load_from_hdf5 \
  --precomp_lang_embed \
  --report_to=wandb
