#!/usr/bin/env bash
# Shared helpers sourced by every script in this directory.
# Not meant to be executed directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

cd "${REPO_ROOT}"

log()  { printf '\033[1;34m[sesaml]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[sesaml]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[sesaml]\033[0m %s\n' "$*" >&2; exit 1; }

# Loads .env (HF_TOKEN, MODEL_REPO_ID, ...) without echoing secret values.
load_env() {
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
    log "Loaded environment from .env"
  else
    warn "No .env found - copy env.template to .env if you need HF credentials"
  fi
}

# Activates .venv when present so scripts work with or without an active shell venv.
activate_venv() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    log "Using active virtualenv: ${VIRTUAL_ENV}"
  elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    log "Activated virtualenv: ${VENV_DIR}"
  elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/Scripts/activate"
    log "Activated virtualenv: ${VENV_DIR}"
  else
    warn "No virtualenv found at ${VENV_DIR} - using system Python (run scripts/setup_env.sh to create one)"
  fi

  PYTHON="${PYTHON:-python3}"
  command -v "${PYTHON}" >/dev/null 2>&1 || die "Python interpreter '${PYTHON}' not found"
  export PYTHON
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}

# Standard bootstrap: repo root + env + interpreter.
bootstrap() {
  load_env
  activate_venv
  mkdir -p "${REPO_ROOT}/outputs/logs" "${REPO_ROOT}/outputs/runs"
}

# Prints the newest run directory of a given kind, e.g. `latest_run train`.
latest_run() {
  local kind="${1:-}"
  local runs_dir="${REPO_ROOT}/outputs/runs"
  [[ -d "${runs_dir}" ]] || return 1
  ls -1d "${runs_dir}/${kind}"*/ 2>/dev/null | sort | tail -n 1
}
