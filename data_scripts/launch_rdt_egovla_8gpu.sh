#!/usr/bin/env bash
set -euo pipefail

# Fresh 8-GPU RDT-1B fine-tune for the validated EgoVLA canonical conversion.
# The W&B API key is intentionally required at runtime and is never stored here.
WORKSPACE="${RDT_EGOVLA_WORKSPACE:-/personal/xiangpc/0812_Xpolicylab_bench/RDT-1B}"
ADAPTER="${WORKSPACE}/Xpolicylab/policy/RDT_1B"
STAGE_ROOT="${RDT_EGOVLA_STAGE_ROOT:-${WORKSPACE}/data/EgoVLA_rdt38}"
DATA_TAG="EgoVLA_benchmark-cotrain-ego_h1_inspire-joint"
DATA_DIR="${ADAPTER}/data/${DATA_TAG}"
LANG_ROOT="${ADAPTER}/lang_embeds"
STATS_PATH="${RDT_EGOVLA_STATS_PATH:-${STAGE_ROOT}/dataset_stat.json}"
OUTPUT_DIR="${RDT_EGOVLA_OUTPUT_DIR:-${WORKSPACE}/chpt/20260902_egovla_joint38_bs64_s42_80k}"
OFFICIAL_OUTPUT="${ADAPTER}/checkpoints/${DATA_TAG}-42"
GPU_IDS="${RDT_GPU_IDS:-0,1,2,3,4,5,6,7}"

to_abs() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$WORKSPACE" "$1" ;;
  esac
}
STAGE_ROOT="$(to_abs "${STAGE_ROOT}")"
STATS_PATH="$(to_abs "${STATS_PATH}")"
OUTPUT_DIR="$(to_abs "${OUTPUT_DIR}")"

die() {
  echo "[RDT-1B][EgoVLA] ERROR: $*" >&2
  exit 1
}

[[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_API_KEY must be exported in the launching shell (not written to this script)"
[[ -d "${STAGE_ROOT}" ]] || die "missing staged data root: ${STAGE_ROOT}"
[[ -f "${STAGE_ROOT}/conversion_manifest.json" ]] || die "missing conversion manifest"
[[ -d "${ADAPTER}" ]] || die "missing RDT adapter: ${ADAPTER}"
[[ -f "${WORKSPACE}/Xpolicylab/utils/get_action_dim.sh" ]] || die "missing robot-dimension helper"
[[ -f "${WORKSPACE}/env_cfg/ego_h1_inspire.yml" ]] || die "missing EgoVLA env config"
[[ -f "${WORKSPACE}/env_cfg/robot/_robot_info.json" ]] || die "missing EgoVLA robot registry"

bash "${WORKSPACE}/Xpolicylab/utils/get_action_dim.sh" "${WORKSPACE}" ego_h1_inspire | tail -n 1 | \
  grep -qx '38' || die "ego_h1_inspire action dimension is not 38"

MANIFEST_CHECK="$(
  "${RDT_EGOVLA_PYTHON:-/personal/miniconda3/envs/rdt_1b/bin/python}" - "${STAGE_ROOT}/conversion_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if m.get("episode_count") != 1903:
    raise SystemExit(f"episode_count={m.get('episode_count')}")
if m.get("raw_inventory", {}).get("deprecated_episode_count") != 100:
    raise SystemExit("deprecated inventory is not 100")
if m.get("action_dim") != 38 or m.get("state_token_dim") != 128:
    raise SystemExit("dimension contract mismatch")
print("ok")
PY
)" || die "invalid staging manifest"
[[ "${MANIFEST_CHECK}" == "ok" ]] || die "staging manifest check failed"

episode_count="$(find -L "${STAGE_ROOT}" -type f -name 'episode_*.hdf5' | wc -l | tr -d ' ')"
[[ "${episode_count}" == "1903" ]] || die "staged episode count=${episode_count}, expected 1903"

[[ -e "${DATA_DIR}" ]] || die "missing process_data link: ${DATA_DIR}"
[[ "$(readlink -f "${DATA_DIR}")" == "$(readlink -f "${STAGE_ROOT}")" ]] || \
  die "RDT data link does not point to the staged tree"
linked_count="$(find -L "${DATA_DIR}" -type f -name 'episode_*.hdf5' | wc -l | tr -d ' ')"
[[ "${linked_count}" == "1903" ]] || die "linked episode count=${linked_count}, expected 1903"

embed_count="$(find -L "${LANG_ROOT}/${DATA_TAG}" -type f -path '*/ego_h1_inspire/lang_embed.pt' | wc -l | tr -d ' ')"
[[ "${embed_count}" == "12" ]] || die "language embedding count=${embed_count}, expected 12"
[[ -s "${LANG_ROOT}/empty_lang_embed.pt" ]] || die "missing empty_lang_embed.pt"

[[ -f "${STATS_PATH}" ]] || die "missing EgoVLA dataset statistics: ${STATS_PATH}"
"${RDT_EGOVLA_PYTHON:-/personal/miniconda3/envs/rdt_1b/bin/python}" - "${STATS_PATH}" <<'PY'
import json
import math
import sys
from pathlib import Path
stats = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = stats.get("egovla_h1_hdf5")
if not isinstance(entry, dict):
    raise SystemExit("stats has no egovla_h1_hdf5 entry")
for field in ("state_mean", "state_std", "state_min", "state_max"):
    values = entry.get(field)
    if not isinstance(values, list) or len(values) != 128:
        raise SystemExit(f"{field} is not length 128")
    if not all(math.isfinite(float(x)) for x in values):
        raise SystemExit(f"{field} contains non-finite values")
active = list(range(0, 7)) + list(range(10, 22)) + list(range(50, 57)) + list(range(60, 72))
std = [float(x) for x in entry["state_std"]]
nonzero = [i for i, x in enumerate(std) if abs(x) > 1e-8]
if set(nonzero) != set(active):
    raise SystemExit(f"stats active slots={nonzero}, expected={active}")
print("stats_ok")
PY

for model_dir in t5-v1_1-xxl siglip-so400m-patch14-384 rdt-1b; do
  [[ -d "${WORKSPACE}/pretrain_model/${model_dir}" ]] || die "missing pretrained model: ${model_dir}"
done

gpu_count="$(tr ',' '\n' <<< "${GPU_IDS}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
[[ "${gpu_count}" == "8" ]] || die "RDT_GPU_IDS must name exactly 8 GPUs (got ${gpu_count})"
if [[ "${RDT_ALLOW_GPU_PROCESSES:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | \
    awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print}')"
  [[ -z "${gpu_apps}" ]] || die "GPU compute processes already running (PIDs: ${gpu_apps}); set RDT_ALLOW_GPU_PROCESSES=1 only after review"
fi

# Never reuse an old checkpoint directory for this fresh run.
mkdir -p "${WORKSPACE}/chpt" "${ADAPTER}/checkpoints"
if [[ -e "${OFFICIAL_OUTPUT}" || -L "${OFFICIAL_OUTPUT}" ]]; then
  if [[ ! -L "${OFFICIAL_OUTPUT}" ]]; then
    die "refusing to replace non-symlink checkpoint path: ${OFFICIAL_OUTPUT}"
  fi
  official_target="$(readlink "${OFFICIAL_OUTPUT}")"
  official_resolved="$(cd "$(dirname "${OFFICIAL_OUTPUT}")" && realpath -m "${official_target}")"
  output_resolved="$(realpath -m "${OUTPUT_DIR}")"
  [[ "${official_resolved}" == "${output_resolved}" ]] || \
    die "checkpoint symlink already points elsewhere: ${OFFICIAL_OUTPUT}"
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  [[ -z "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
    die "fresh output directory is not empty: ${OUTPUT_DIR}"
else
  mkdir -p "${OUTPUT_DIR}"
fi
if [[ ! -e "${OFFICIAL_OUTPUT}" && ! -L "${OFFICIAL_OUTPUT}" ]]; then
  ln -s "${OUTPUT_DIR}" "${OFFICIAL_OUTPUT}"
fi

# Explicit model paths avoid the repository's small placeholder weight links.
export PATH="/personal/miniconda3/envs/rdt_1b/bin:${PATH}"
export TEXT_ENCODER_NAME="${WORKSPACE}/pretrain_model/t5-v1_1-xxl"
export VISION_ENCODER_NAME="${WORKSPACE}/pretrain_model/siglip-so400m-patch14-384"
export RDT_PRETRAINED_MODEL="${WORKSPACE}/pretrain_model/rdt-1b"
export RDT_HDF5_DIR="${DATA_DIR}"
export RDT_LANG_EMBED_DIR="${LANG_ROOT}"
export RDT_DATASET_NAME="egovla_h1_hdf5"
export RDT_DATASET_STAT_PATH="${STATS_PATH}"
export RDT_DROP_SHORT_EPISODES=0
export RDT_TRAIN_BATCH_SIZE=8
export RDT_SAMPLE_BATCH_SIZE=8
export RDT_MAX_TRAIN_STEPS=80000
export RDT_CHECKPOINTING_PERIOD=10000
export RDT_SAMPLE_PERIOD="${RDT_SAMPLE_PERIOD:-0}"
export RDT_CHECKPOINTS_TOTAL_LIMIT=40
export RDT_DATALOADER_NUM_WORKERS="${RDT_DATALOADER_NUM_WORKERS:-4}"
export RDT_RESUME_FROM_CHECKPOINT=""
export RDT_REPORT_TO="${RDT_REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-xpolicylab-0812-bench}"
export WANDB_NAME="${WANDB_NAME:-RDT_1B_EgoVLA_joint38_bs64_s42_80k_20260902}"
export http_proxy="${http_proxy:-http://192.168.16.76:18000}"
export https_proxy="${https_proxy:-http://192.168.16.76:18000}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[RDT-1B][EgoVLA] stage=${STAGE_ROOT} episodes=1903 deprecated_excluded=100"
echo "[RDT-1B][EgoVLA] data_tag=${DATA_TAG} dataset=egovla_h1_hdf5"
echo "[RDT-1B][EgoVLA] raw_action_dim=38 mapped_state_tokens=128 ctrl_freq=30"
echo "[RDT-1B][EgoVLA] train_batch_per_gpu=8 sample_batch_per_gpu=8 global_batch=64"
echo "[RDT-1B][EgoVLA] max_steps=80000 checkpoint_period=10000 sample_period=${RDT_SAMPLE_PERIOD}"
echo "[RDT-1B][EgoVLA] output=${OUTPUT_DIR}"
echo "[RDT-1B][EgoVLA] WANDB_PROJECT=${WANDB_PROJECT} WANDB_NAME=${WANDB_NAME} WANDB_API_KEY=present"
echo "[RDT-1B][EgoVLA] launching 8 ranks on CUDA_VISIBLE_DEVICES=${GPU_IDS}"

cd "${ADAPTER}"
exec bash train.sh EgoVLA_benchmark cotrain ego_h1_inspire joint 42 "${GPU_IDS}"
