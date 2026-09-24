#!/usr/bin/env bash
set -euo pipefail

eval_batch="${1}"
eval_env_conda_env="${2}"
free_port="${3}"
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

policy_root="${XPOLICYLAB_ROOT:-${root_dir}/XPolicyLab}"
debug_client="${policy_root}/debug_env_client.py"
if [[ ! -f "${debug_client}" ]]; then
    echo "[ERROR] debug client is missing: ${debug_client}" >&2
    exit 1
fi
debug_python="$(xpl_python_for_env "${eval_env_conda_env}" 2>/dev/null || true)"
if [[ -z "${debug_python}" || ! -x "${debug_python}" ]]; then
    echo "[ERROR] no Python interpreter for debug environment ${eval_env_conda_env@Q}" >&2
    exit 1
fi

export PYTHONPATH="${policy_root%/XPolicyLab}:${policy_root}${PYTHONPATH:+:${PYTHONPATH}}"
echo -e "\033[34m[CLIENT] Debug client Python: ${debug_python}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${free_port}...\033[0m"

"${debug_python}" "${debug_client}" \
    --bench_name "${bench_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --protocol "${protocol}" \
    --host "${policy_server_ip}" \
    --port "${free_port}" \
    --eval_batch "${eval_batch}"
