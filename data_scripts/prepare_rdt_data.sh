#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_ROOT="${SPARK0_HDF5_ROOT:-/personal/tjy/spark0_bench}"

bash "${SCRIPT_DIR}/link_raw_hdf5.sh"
cd "${POLICY_ROOT}/Xpolicylab/policy/RDT_1B"
exec bash process_data.sh \
  Spark0_bench cotrain tianji_marvin_wuji joint \
  "${SOURCE_ROOT}" "$@"
