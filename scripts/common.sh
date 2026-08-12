#!/usr/bin/env bash
# Shared helpers sourced by every script under scripts/.
# Not meant to be executed directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Walk up to the repository root rather than assuming a fixed depth: scripts
# live in subdirectories (asr/, datasets/, lm/, ...), and a hard-coded `..` was
# only ever correct while they all sat directly in scripts/.
REPO_ROOT="${SCRIPT_DIR}"
while [[ "${REPO_ROOT}" != "/" && ! -f "${REPO_ROOT}/requirements.txt" ]]; do
  REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
[[ -f "${REPO_ROOT}/requirements.txt" ]] || {
  printf '\033[1;31m[sesaml]\033[0m Could not locate the repository root from %s\n' "${SCRIPT_DIR}" >&2
  exit 1
}

SCRIPTS_DIR="${REPO_ROOT}/scripts"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

cd "${REPO_ROOT}"

log()  { printf '\033[1;34m[sesaml]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[sesaml]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[sesaml]\033[0m %s\n' "$*" >&2; exit 1; }

# Loads .env (HF_TOKEN, MODEL_REPO_ID, ...) without echoing secret values.
#
# Parsed line by line rather than sourced. `source .env` executes the file as a
# shell script, so any stray line runs as a command - a pasted
# `python scripts/datasets/download_all_datasets.py ...` left in .env re-downloaded every
# corpus on each invocation of every script, before the actual work started.
# Only KEY=VALUE assignments are honoured here, and anything else is reported
# rather than run.
load_env() {
  local env_file="${REPO_ROOT}/.env"

  if [[ ! -f "${env_file}" ]]; then
    warn "No .env found - copy env.template to .env if you need HF credentials"
    return 0
  fi

  local line key value skipped=0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"                       # tolerate CRLF from Windows editors

    if [[ -z "${line//[[:space:]]/}" ]]; then
      continue
    fi
    if [[ "${line}" =~ ^[[:space:]]*# ]]; then
      continue
    fi

    if [[ "${line}" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[2]}"
      value="${BASH_REMATCH[3]}"
      # Strip one layer of matching quotes, as dotenv files conventionally allow.
      if [[ "${value}" =~ ^\"(.*)\"$ ]] || [[ "${value}" =~ ^\'(.*)\'$ ]]; then
        value="${BASH_REMATCH[1]}"
      fi
      export "${key}=${value}"
    else
      skipped=$((skipped + 1))
    fi
  done < "${env_file}"

  if (( skipped > 0 )); then
    # Deliberately does not echo the line: it may carry a token.
    warn ".env: ignored ${skipped} line(s) that are not KEY=VALUE assignments."
    warn "      Commands in .env are no longer executed. Remove them, or move them to a script."
  fi

  log "Loaded environment from .env"
}

# Finds or installs a compatible Python interpreter (>=3.10, <3.14).
detect_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
    return
  fi

  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "${VENV_DIR}/bin/python"
    return
  fi

  for candidate in python3.11 python3.12 python3.13 python3.10 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1; then
        echo "${candidate}"
        return
      fi
    fi
  done

  if [[ -x "${SCRIPTS_DIR}/setup/ensure_python.sh" ]]; then
    "${SCRIPTS_DIR}/setup/ensure_python.sh"
  else
    echo "python3"
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
    warn "No virtualenv found at ${VENV_DIR} - using system Python (run scripts/setup/setup_env.sh to create one)"
  fi

  PYTHON="$(detect_python)"
  command -v "${PYTHON}" >/dev/null 2>&1 || die "Python interpreter '${PYTHON}' not found"
  export PYTHON
  export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}

# Standard bootstrap: repo root + env + interpreter.
bootstrap() {
  load_env
  activate_venv
  mkdir -p "${REPO_ROOT}/outputs/logs" "${REPO_ROOT}/outputs/asr/runs"
}

# Prints the newest run directory of a given kind, e.g. `latest_run train`.
latest_run() {
  local kind="${1:-}"
  local runs_dir="${REPO_ROOT}/outputs/asr/runs"
  [[ -d "${runs_dir}" ]] || return 1
  ls -1d "${runs_dir}/${kind}"*/ 2>/dev/null | sort | tail -n 1
}
