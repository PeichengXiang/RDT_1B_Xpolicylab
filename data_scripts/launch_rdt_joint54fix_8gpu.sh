#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/personal/xiangpc/0812_Xpolicylab_bench/RDT-1B"
ADAPTER="${WORKSPACE}/Xpolicylab/policy/RDT_1B"
OUTPUT_DIR="${WORKSPACE}/chpt/20260813_mnt_joint54fix_bs64_s42_80k"
OFFICIAL_OUTPUT="${ADAPTER}/checkpoints/Spark0_bench-cotrain-tianji_marvin_wuji-joint-42"

[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }
for model_dir in t5-v1_1-xxl siglip-so400m-patch14-384 rdt-1b; do
  [[ -d "${WORKSPACE}/pretrain_model/${model_dir}" ]] || { echo "Missing ${model_dir}" >&2; exit 1; }
done
[[ -e "${ADAPTER}/lang_embeds/Spark0_bench-cotrain-tianji_marvin_wuji-joint" ]] || {
  echo "Missing RDT language embeddings" >&2
  exit 1
}

mkdir -p "${WORKSPACE}/chpt" "${ADAPTER}/checkpoints"
if [[ -e "${OFFICIAL_OUTPUT}" && ! -L "${OFFICIAL_OUTPUT}" ]]; then
  echo "Refusing to replace non-symlink output: ${OFFICIAL_OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"
ln -sfn "${OUTPUT_DIR}" "${OFFICIAL_OUTPUT}"

export PATH="/personal/miniconda3/envs/rdt_1b/bin:${PATH}"
export TEXT_ENCODER_NAME="${WORKSPACE}/pretrain_model/t5-v1_1-xxl"
export VISION_ENCODER_NAME="${WORKSPACE}/pretrain_model/siglip-so400m-patch14-384"
export RDT_PRETRAINED_MODEL="${WORKSPACE}/pretrain_model/rdt-1b"
export RDT_TRAIN_BATCH_SIZE=8
export RDT_MAX_TRAIN_STEPS=80000
export RDT_CHECKPOINTING_PERIOD=10000
export RDT_RESUME_FROM_CHECKPOINT=""
export RDT_CHECKPOINTS_TOTAL_LIMIT=40
export RDT_REPORT_TO=wandb
export WANDB_PROJECT=xpolicylab-0812-bench
export WANDB_NAME=RDT_1B_mnt_joint54fix_bs64_s42_80k_fresh_20260813
export http_proxy="${http_proxy:-http://192.168.16.76:18000}"
export https_proxy="${https_proxy:-http://192.168.16.76:18000}"

exec bash "${ADAPTER}/train.sh" Spark0_bench cotrain tianji_marvin_wuji joint 42 0,1,2,3,4,5,6,7
