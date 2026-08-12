#!/usr/bin/env bash
# Creates the project virtualenv and installs dependencies.
#
# Usage: scripts/setup_env.sh [--recreate]

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RECREATE=0
for arg in "$@"; do
  case "${arg}" in
    --recreate) RECREATE=1 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) die "Unknown argument: ${arg}" ;;
  esac
done

PYTHON="$(detect_python)"

if [[ "${RECREATE}" -eq 1 && -d "${VENV_DIR}" ]]; then
  log "Removing existing virtualenv at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating virtualenv at ${VENV_DIR}"
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Upgrading pip"
python -m pip install --upgrade pip >/dev/null

log "Installing requirements.txt"
python -m pip install -r "${REPO_ROOT}/requirements.txt"

mkdir -p "${REPO_ROOT}/outputs/logs" \
         "${REPO_ROOT}/outputs/runs" \
         "${REPO_ROOT}/outputs/checkpoints" \
         "${REPO_ROOT}/outputs/exports" \
         "${REPO_ROOT}/data"

if [[ ! -f "${REPO_ROOT}/.env" && -f "${REPO_ROOT}/env.template" ]]; then
  cp "${REPO_ROOT}/env.template" "${REPO_ROOT}/.env"
  log "Created .env from env.template - add your HF_TOKEN there"
fi

log "Environment ready. Activate it with: source ${VENV_DIR}/bin/activate"
