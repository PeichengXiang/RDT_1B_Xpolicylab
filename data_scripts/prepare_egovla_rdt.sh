#!/usr/bin/env bash
set -euo pipefail

# Prepare the official EgoVLA canonical conversion for RDT-1B. This command
# only creates links under the model root and never copies the large RGB data.
WORKSPACE="${RDT_EGOVLA_WORKSPACE:-/personal/xiangpc/0812_Xpolicylab_bench/RDT-1B}"
ADAPTER="${WORKSPACE}/Xpolicylab/policy/RDT_1B"
SCRIPT_DIR="${WORKSPACE}/data_scripts"
STAGE_ROOT="${RDT_EGOVLA_STAGE_ROOT:-${WORKSPACE}/data/EgoVLA_rdt38}"
DATA_TAG="EgoVLA_benchmark-cotrain-ego_h1_inspire-joint"
DATA_LINK="${ADAPTER}/data/${DATA_TAG}"
STATS_PATH="${RDT_EGOVLA_STATS_PATH:-${STAGE_ROOT}/dataset_stat.json}"
ENCODE_GPU=0
STAGE_ONLY=0
SKIP_FILE_VALIDATION=0
SKIP_ENCODE=0
SKIP_STATS=0

usage() {
  cat <<'EOF'
Usage: prepare_egovla_rdt.sh [options]

Stages the 1,903 active EgoVLA canonical episodes under model-root/data,
links that tree into policy/RDT_1B/data, pre-encodes one instruction per
task, and computes a 128-slot dataset-stat file.

Options:
  --gpu N                 GPU used for T5 language encoding (default: 0)
  --skip-file-validation  Skip per-HDF5 schema scans (inventory/count checks remain)
  --skip-encode           Do not run process_data.sh language encoding
  --skip-stats            Do not compute dataset statistics
  --stage-only            Stop after creating/verifying model-root/data
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) ENCODE_GPU="${2:?missing GPU id}"; shift ;;
    --skip-file-validation) SKIP_FILE_VALIDATION=1 ;;
    --skip-encode) SKIP_ENCODE=1 ;;
    --skip-stats) SKIP_STATS=1 ;;
    --stage-only) STAGE_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

RAW_ROOT="/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/raw"
CANONICAL_ROOT="/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/canonical"
PYTHON_BIN="${RDT_EGOVLA_PYTHON:-/personal/miniconda3/envs/rdt_1b/bin/python}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${RAW_ROOT}" ]] || { echo "Missing raw root: ${RAW_ROOT}" >&2; exit 1; }
[[ -d "${CANONICAL_ROOT}" ]] || { echo "Missing canonical root: ${CANONICAL_ROOT}" >&2; exit 1; }
[[ -x "${SCRIPT_DIR}/prepare_egovla_rdt.py" ]] || {
  echo "Missing staging helper: ${SCRIPT_DIR}/prepare_egovla_rdt.py" >&2
  exit 1
}

if [[ -e "${STAGE_ROOT}" ]]; then
  [[ -f "${STAGE_ROOT}/conversion_manifest.json" ]] || {
    echo "Existing stage has no conversion_manifest.json: ${STAGE_ROOT}" >&2
    exit 1
  }
  count="$(find -L "${STAGE_ROOT}" -type f -name '*.hdf5' | wc -l | tr -d ' ')"
  [[ "${count}" == "1903" ]] || {
    echo "Existing stage has ${count} HDF5 files; expected 1903" >&2
    exit 1
  }
  "${PYTHON_BIN}" - "${STAGE_ROOT}/conversion_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("episode_count") != 1903:
    raise SystemExit(f"stage manifest episode_count={manifest.get('episode_count')}, expected 1903")
if manifest.get("raw_inventory", {}).get("deprecated_episode_count") != 100:
    raise SystemExit("stage manifest does not record exactly 100 excluded Deprecated episodes")
if manifest.get("action_dim") != 38 or manifest.get("state_token_dim") != 128:
    raise SystemExit("stage manifest has an unexpected EgoVLA dimension contract")
print("STAGE_MANIFEST_OK active=1903 deprecated_excluded=100")
PY
else
  stage_args=(--raw-root "${RAW_ROOT}" --canonical-root "${CANONICAL_ROOT}" --output-root "${STAGE_ROOT}")
  if [[ "${SKIP_FILE_VALIDATION}" == "1" ]]; then
    stage_args+=(--skip-file-validation)
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_egovla_rdt.py" "${stage_args[@]}"
fi

declare -A EXPECTED=(
  [Close-Drawer]=50 [Flip-Mug]=100 [Insert-And-Unload-Cans]=900
  [Insert-Cans]=100 [Open-Drawer]=100 [Open-Laptop]=100 [Pour-Balls]=102
  [Push-Box]=100 [Sort-Cans]=101 [Stack-Can]=100
  [Stack-Can-Into-Drawer]=50 [Unload-Cans]=100
)
for task in "${!EXPECTED[@]}"; do
  directory="${STAGE_ROOT}/${task}/ego_h1_inspire/data"
  count="$(find -L "${directory}" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l | tr -d ' ')"
  [[ "${count}" == "${EXPECTED[$task]}" ]] || {
    echo "${task}: expected ${EXPECTED[$task]} episodes, found ${count}" >&2
    exit 1
  }
done

if [[ "${SKIP_ENCODE}" == "1" && ! -e "${DATA_LINK}" ]]; then
  mkdir -p "${ADAPTER}/data"
  ln -s "${STAGE_ROOT}" "${DATA_LINK}"
fi

if [[ "${STAGE_ONLY}" == "1" ]]; then
  echo "STAGE_OK root=${STAGE_ROOT} episodes=1903"
  exit 0
fi

# process_data.sh needs conda to activate the RDT environment for the T5 model.
# A conda executable can be on PATH without its shell function being initialized
# (the common case for non-interactive SSH shells), so source the profile
# unconditionally when it is available.
if [[ -f /personal/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /personal/miniconda3/etc/profile.d/conda.sh
elif ! command -v conda >/dev/null 2>&1; then
  echo "conda is unavailable; cannot run process_data.sh" >&2
  exit 1
fi
conda activate "${RDT_CONDA_ENV:-rdt_1b}"

if [[ "${SKIP_ENCODE}" != "1" ]]; then
  mkdir -p "${ADAPTER}/data" "${ADAPTER}/lang_embeds"
  cd "${ADAPTER}"
  bash process_data.sh EgoVLA_benchmark cotrain ego_h1_inspire joint "${STAGE_ROOT}" --gpu "${ENCODE_GPU}"
fi

if [[ "${SKIP_STATS}" != "1" ]]; then
  [[ -d "${DATA_LINK}" ]] || {
    echo "Missing linked RDT data tree: ${DATA_LINK}; run without --skip-encode" >&2
    exit 1
  }
  if [[ ! -f "${STATS_PATH}" ]]; then
    export RDT_HDF5_DIR="${DATA_LINK}"
    export RDT_DATASET_NAME=egovla_h1_hdf5
    export RDT_DROP_SHORT_EPISODES=0
    cd "${ADAPTER}/rdt"
    mkdir -p "$(dirname "${STATS_PATH}")"
    PYTHONPATH=. "${PYTHON_BIN}" data/compute_dataset_stat_hdf5.py --save_path "${STATS_PATH}"
  fi
  "${PYTHON_BIN}" - "${STATS_PATH}" <<'PY'
import json
import sys
from pathlib import Path
stats = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = stats.get("egovla_h1_hdf5")
if entry is None:
    raise SystemExit("stats file has no egovla_h1_hdf5 entry")
for field in ("state_mean", "state_std", "state_min", "state_max"):
    if len(entry.get(field, [])) != 128:
        raise SystemExit(f"{field} length is not 128")
print("STATS_OK dataset=egovla_h1_hdf5 state_dim=128")
PY
fi

echo "PREPARE_OK stage=${STAGE_ROOT} data_link=${DATA_LINK} episodes=1903 deprecated_excluded=100"
