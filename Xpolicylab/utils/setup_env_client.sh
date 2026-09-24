#!/bin/bash
set -e

UTILS_DIR="${1}"
yaml_file="${2}"
eval_env_conda_env="${3}"
policy_server_port="${4}"
bench_name="${5}"
task_name="${6}"
env_cfg_type="${7}"
policy_name="${8}"
additional_info="${9}"
ROOT_DIR="${10}"
seed="${11}"
env_gpu_id="${12}"
policy_server_ip="${13:-localhost}"
protocol_override="${14:-}"

# Resolve the interpreter before the environment-specific client starts.  A
# non-interactive SSH shell on the benchmark host has neither a `python`
# alias nor an initialized Conda function, so relying on either here makes
# evaluation fail before the simulator/debug client is reached.
# shellcheck source=conda_env.sh
source "${UTILS_DIR}/conda_env.sh"
yaml_python="${XPL_YAML_PYTHON:-}"
if [[ -z "${yaml_python}" ]]; then
    yaml_python="$(xpl_python_for_env "${eval_env_conda_env}" 2>/dev/null || true)"
fi
if [[ -z "${yaml_python}" ]]; then
    xpl_activate_env "${eval_env_conda_env}" 2>/dev/null || true
    yaml_python="$(xpl_python_for_env "${eval_env_conda_env}" 2>/dev/null || true)"
fi
if [[ -z "${yaml_python}" || ! -x "${yaml_python}" ]]; then
    echo "[ERROR] cannot find a Python interpreter for eval environment ${eval_env_conda_env@Q}; set XPL_YAML_PYTHON" >&2
    exit 1
fi
export XPL_YAML_PYTHON="${yaml_python}"

# shellcheck source=resolve_eval_env_type.sh
source "${UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)" || exit 1

read eval_batch yaml_protocol < <("${yaml_python}" - "${yaml_file}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
print(
    str(data.get("eval_batch", False)).lower(),
    data.get("protocol", "ws"),
)
PY
)
protocol="${protocol_override:-${yaml_protocol}}"
eval_batch="${RDT_EVAL_BATCH:-${eval_batch}}"
case "${eval_batch,,}" in
    true|false) eval_batch="${eval_batch,,}" ;;
    *) echo "[ERROR] RDT_EVAL_BATCH must be true or false (got ${eval_batch@Q})" >&2; exit 2 ;;
esac

if [[ -z "${EVAL_ENV_TYPE:-}" ]]; then
    echo "[CLIENT] EVAL_ENV_TYPE=(default sim) -> ${eval_env_mode}"
else
    echo "[CLIENT] EVAL_ENV_TYPE=${EVAL_ENV_TYPE} -> ${eval_env_mode}"
fi

if [[ "${RDT_EVAL_DRY_RUN:-0}" == "1" ]]; then
    echo "[CLIENT][DRY-RUN] mode=${eval_env_mode}, protocol=${protocol}, eval_batch=${eval_batch}"
    echo "[CLIENT][DRY-RUN] root=${ROOT_DIR}, server=${policy_server_ip}:${policy_server_port}"
    exit 0
fi

COMMON_ARGS=(
    "${eval_batch}"
    "${eval_env_conda_env}"
    "${policy_server_port}"
    "${bench_name}"
    "${task_name}"
    "${env_cfg_type}"
    "${policy_name}"
    "${additional_info}"
    "${ROOT_DIR}"
    "${seed}"
    "${env_gpu_id}"
    "${policy_server_ip}"
)

if [[ "${eval_env_mode}" == "debug" ]]; then
    bash "${UTILS_DIR}/run_debug_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
elif [[ "${eval_env_mode}" == "sim" ]]; then
    bash "${UTILS_DIR}/run_sim_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
elif [[ "${eval_env_mode}" == "real_world" ]]; then
    echo -e "\033[31m[WARN] EVAL_ENV_TYPE=real: real-world evaluation is not supported in the open-source release; continuing to real env client.\033[0m" >&2
    bash "${UTILS_DIR}/run_real_env_client.sh" "${COMMON_ARGS[@]}" "${protocol}"
else
    echo "[ERROR] Unknown eval env mode: ${eval_env_mode}" >&2
    exit 1
fi
