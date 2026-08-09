#!/usr/bin/env bash
# Runs the test suite. Extra flags are forwarded to pytest.
#
# Usage:
#   scripts/run_tests.sh
#   scripts/run_tests.sh tests/test_model.py -k forward

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bootstrap

if "${PYTHON}" -c "import pytest" >/dev/null 2>&1; then
  log "Running pytest"
  exec "${PYTHON}" -m pytest "${@:-tests}"
fi

warn "pytest not installed - falling back to tests/run_all_tests.py"
exec "${PYTHON}" "${REPO_ROOT}/tests/run_all_tests.py"
