#!/usr/bin/env bash
# Checks if a compatible Python (>=3.10, <3.14) is installed.
# If none is found, installs Python 3.11 automatically using uv or apt.
#
# Usage:
#   scripts/ensure_python.sh
#   PYTHON_BIN=$(scripts/ensure_python.sh)

set -euo pipefail

is_compatible() {
  local bin="$1"
  command -v "${bin}" >/dev/null 2>&1 || return 1
  "${bin}" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1
}

find_python() {
  # 1. Respect explicit PYTHON override if valid
  if [[ -n "${PYTHON:-}" ]] && is_compatible "${PYTHON}"; then
    command -v "${PYTHON}" || echo "${PYTHON}"
    return 0
  fi

  # 2. Check candidate binaries in PATH and standard install paths
  for candidate in \
    python3.11 python3.12 python3.13 python3.10 python3 \
    /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 \
    "${HOME}/.local/share/uv/python/cpython-3.11"*"/bin/python3"; do
    if [[ -n "${candidate}" ]] && is_compatible "${candidate}"; then
      command -v "${candidate}" || echo "${candidate}"
      return 0
    fi
  done
  return 1
}

install_with_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    if command -v curl >/dev/null 2>&1; then
      printf '\033[1;34m[sesaml]\033[0m Installing uv package manager...\n' >&2
      curl -LsSf https://astral.sh/uv/install.sh | sh >&2 || return 1
      export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    else
      return 1
    fi
  fi

  if command -v uv >/dev/null 2>&1; then
    printf '\033[1;34m[sesaml]\033[0m Installing Python 3.11 via uv...\n' >&2
    uv python install 3.11 >&2
    local uv_py
    uv_py="$(uv python find 3.11 2>/dev/null || true)"
    if [[ -n "${uv_py}" ]] && is_compatible "${uv_py}"; then
      echo "${uv_py}"
      return 0
    fi
  fi
  return 1
}

install_with_apt() {
  if command -v apt-get >/dev/null 2>&1; then
    printf '\033[1;34m[sesaml]\033[0m Installing Python 3.11 via apt-get...\n' >&2
    sudo apt-get update -y >&2
    sudo apt-get install -y software-properties-common >&2
    sudo add-apt-repository ppa:deadsnakes/ppa -y >&2
    sudo apt-get update -y >&2
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev portaudio19-dev >&2
    if is_compatible python3.11; then
      echo "python3.11"
      return 0
    fi
  fi
  return 1
}

MAIN_PY=""
if MAIN_PY=$(find_python); then
  echo "${MAIN_PY}"
  exit 0
fi

printf '\033[1;33m[sesaml]\033[0m No compatible Python (>=3.10, <3.14) found. Attempting auto-installation...\n' >&2

if MAIN_PY=$(install_with_uv); then
  echo "${MAIN_PY}"
  exit 0
fi

if MAIN_PY=$(install_with_apt); then
  echo "${MAIN_PY}"
  exit 0
fi

printf '\033[1;31m[sesaml]\033[0m Failed to automatically install Python 3.11. Please install Python 3.11 manually.\n' >&2
exit 1
