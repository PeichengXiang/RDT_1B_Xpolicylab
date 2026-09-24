#!/usr/bin/env bash
set -euo pipefail

eval_batch="${1}"
eval_env_conda_env="${2}"
policy_server_port="${3}"
bench_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"
protocol="${13:-ws}"

UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=conda_env.sh
source "${UTILS_DIR}/conda_env.sh"
if ! xpl_activate_env "${eval_env_conda_env}"; then
    echo "[ERROR] cannot activate eval environment ${eval_env_conda_env@Q}; set XPL_CONDA_SH/RDT_CONDA_BASE or pass an absolute prefix" >&2
    exit 1
fi

if [[ ! -f "${root_dir}/scripts/eval_policy.sh" ]]; then
    echo "[ERROR] benchmark eval hook is missing: ${root_dir}/scripts/eval_policy.sh (set EVAL_MAIN_ROOT)" >&2
    exit 1
fi

echo -e "\033[34m[CLIENT] Activating eval environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means retrying.\033[0m"

# The benchmark bridge accepts the public suite name EgoVLA, while the RDT
# training data tag intentionally contains the `_benchmark` suffix. Keep the
# training/server name untouched but normalize only the bridge argument.
bridge_bench_name="${bench_name}"
if [[ "${policy_name}" == "RDT_1B" && "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    bridge_bench_name="${EGOVLA_BRIDGE_BENCH_NAME:-EgoVLA}"
fi
policy_root="${XPOLICYLAB_ROOT:-}"
bridge_args=(
    --bench_name "${bridge_bench_name}"
    --task_name "${task_name}"
    --env_cfg_type "${env_cfg_type}"
    --policy_name "${policy_name}"
    --host "${policy_server_ip}"
    --port "${policy_server_port}"
    --protocol "${protocol}"
    --eval_batch "${eval_batch}"
    --root_dir "${root_dir}"
    --device_id "${env_gpu_id}"
    --additional_info "${additional_info}"
    --seed "${seed}"
)
if [[ -n "${policy_root}" ]]; then
    # Without this explicit path the bridge may import its stale benchmark
    # XPolicyLab checkout instead of the 38D adapter used by the server.
    bridge_args+=(--xpolicy-root "${policy_root}")
fi
# Explicit flags avoid the legacy comma-delimited checkpoint parser and keep
# paths containing commas/spaces lossless.
if [[ -n "${EGOVLA_CHECKPOINT:-}" ]]; then bridge_args+=(--checkpoint "${EGOVLA_CHECKPOINT}"); fi
if [[ -n "${EGOVLA_ACTION_TYPE:-}" ]]; then bridge_args+=(--action-type "${EGOVLA_ACTION_TYPE}"); fi
if [[ -n "${EGOVLA_INSTRUCTION:-}" ]]; then bridge_args+=(--instruction "${EGOVLA_INSTRUCTION}"); fi
if [[ -n "${EGOVLA_BENCHMARK_PROTOCOL:-}" ]]; then bridge_args+=(--benchmark-protocol "${EGOVLA_BENCHMARK_PROTOCOL}"); fi
if [[ "${policy_name}" == "RDT_1B" && "${EGOVLA_RDT_EVAL_FORCE:-0}" == "1" ]]; then
    # The published bridge manifest intentionally denies modified policy trees;
    # this explicit opt-in is limited to the local adapter's eval wrapper.
    bridge_args+=(--force)
fi
if [[ -n "${EGOVLA_CAPABILITY_MANIFEST:-}" ]]; then
    bridge_args+=(--manifest "${EGOVLA_CAPABILITY_MANIFEST}")
fi

if [[ "${RDT_EVAL_DRY_RUN:-0}" == "1" ]]; then
    printf '[CLIENT][DRY-RUN] bridge command:'
    printf ' %q' "${root_dir}/scripts/eval_policy.sh" "${bridge_args[@]}"
    printf '\n'
    exit 0
fi

bash "${root_dir}/scripts/eval_policy.sh" "${bridge_args[@]}"
