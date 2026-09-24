#!/bin/bash
set -e

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
    echo "[ERROR] cannot activate eval environment ${eval_env_conda_env@Q}" >&2
    exit 1
fi
real_python="$(xpl_python_for_env "${eval_env_conda_env}" 2>/dev/null || true)"
[[ -n "${real_python}" && -x "${real_python}" ]] || { echo "[ERROR] no Python for real env" >&2; exit 1; }
IFS='|' read -r action_type base_cfg < <(ADDITIONAL_INFO="${additional_info}" "${real_python}" - <<'PY'
import os

fields = {}
for part in os.environ.get("ADDITIONAL_INFO", "").split(","):
    if "=" not in part:
        continue
    key, value = part.split("=", 1)
    fields[key.strip()] = value.strip()
print(fields.get("action_type", "") + "|" + fields.get("base_cfg", ""))
PY
)
base_cfg="${base_cfg:-${BASE_CFG:-}}"

echo -e "\033[31m[WARN] Real-world evaluation is not supported in the open-source release; attempting real env client startup anyway.\033[0m" >&2

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${free_port} (real env)...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means the client is retrying.\033[0m"

export PYTHONPATH="${root_dir}/src:${root_dir}/XPolicyLab:${root_dir}:${PYTHONPATH:-}"

if [[ -z "${action_type}" ]]; then
    echo "[ERROR] EVAL_ENV_TYPE=real requires action_type (via additional_info or --action_type)" >&2
    exit 1
fi

if [[ -z "${base_cfg}" ]]; then
    echo "[ERROR] EVAL_ENV_TYPE=real requires base_cfg (via additional_info, --base_cfg, or BASE_CFG)" >&2
    exit 1
fi

"${real_python}" "${root_dir}/src/task_env/real_env_client.py" \
    --dataset_name "${bench_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --protocol "${protocol}" \
    --host "${policy_server_ip}" \
    --port "${free_port}" \
    --eval_batch "${eval_batch}" \
    --seed "${seed}" \
    --action_type "${action_type}" \
    --base_cfg "${base_cfg}" \
    --additional_info "${additional_info}" \
    --root_dir "${root_dir}"
