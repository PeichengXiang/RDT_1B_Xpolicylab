#!/usr/bin/env bash
# Shared non-interactive Conda/environment helpers for XPolicyLab launchers.
# This file is sourced by the evaluation scripts; it has no side effects
# until one of the functions below is called.

xpl_source_conda() {
    local candidate base exe

    # Prefer explicitly selected init scripts.  This is useful on hosts where
    # a non-interactive SSH shell does not source conda.sh.
    for candidate in "${XPL_CONDA_SH:-}" "${RDT_CONDA_SH:-}" "${CONDA_SH:-}" \
        "/personal/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -n "${candidate}" && -r "${candidate}" ]]; then
            # shellcheck disable=SC1090
            source "${candidate}"
            return 0
        fi
    done

    # Also accept a configured Conda installation or executable.
    for base in "${XPL_CONDA_BASE:-}" "${RDT_CONDA_BASE:-}" "${CONDA_BASE:-}"; do
        if [[ -n "${base}" && -r "${base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${base}/etc/profile.d/conda.sh"
            return 0
        fi
    done
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        base="$(cd "$(dirname "${CONDA_EXE}")/.." 2>/dev/null && pwd || true)"
        if [[ -n "${base}" && -r "${base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${base}/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    if command -v conda >/dev/null 2>&1; then
        candidate="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${candidate}" && -r "${candidate}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${candidate}/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    return 1
}

xpl_activate_env() {
    local env_ref="${1:-}"
    [[ -n "${env_ref}" ]] || return 1

    # An absolute environment prefix does not need a shell function.  Adding
    # its bin directory makes the selected Python visible to child scripts.
    if [[ -x "${env_ref}/bin/python" ]]; then
        export PATH="${env_ref}/bin${PATH:+:${PATH}}"
        export CONDA_PREFIX="${env_ref}"
        export CONDA_DEFAULT_ENV="${env_ref}"
        export XPL_ACTIVE_ENV="${env_ref}"
        return 0
    fi

    xpl_source_conda || return 1
    command -v conda >/dev/null 2>&1 || return 1
    conda activate "${env_ref}"
    export XPL_ACTIVE_ENV="${env_ref}"
}

xpl_python_for_env() {
    local env_ref="${1:-}" base candidate
    if [[ -x "${env_ref}/bin/python" ]]; then
        printf '%s\n' "${env_ref}/bin/python"
        return 0
    fi
    if [[ -x "${env_ref}/bin/python3" ]]; then
        printf '%s\n' "${env_ref}/bin/python3"
        return 0
    fi
    # Resolve named environments under configured Conda bases before falling
    # back to PATH/base Python. This avoids selecting a dependency-poor base
    # interpreter for a named policy or simulator environment.
    for base in "${XPL_CONDA_BASE:-}" "${RDT_CONDA_BASE:-}" "${CONDA_BASE:-}" "/personal/miniconda3"; do
        [[ -n "${base}" ]] || continue
        candidate="${base}/envs/${env_ref}/bin/python"
        if [[ ! -x "${candidate}" ]]; then
            candidate="${base}/envs/${env_ref}/bin/python3"
        fi
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    # A minimal host may expose only a base interpreter; use it only after
    # checking the requested named environment and the active PATH.
    for base in "${XPL_CONDA_BASE:-}" "${RDT_CONDA_BASE:-}" "${CONDA_BASE:-}" "/personal/miniconda3"; do
        [[ -n "${base}" ]] || continue
        candidate="${base}/bin/python"
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}
