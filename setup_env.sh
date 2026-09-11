#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${PROJECT_ROOT}/../.conda-envs/mbrs"
CONDA_CACHE="/tmp/mbrs-conda-pkgs"

if ! command -v mamba >/dev/null 2>&1; then
  echo "ERROR: mamba is required; install Miniforge or use conda create manually." >&2
  exit 1
fi

mkdir -p "${CONDA_CACHE}" "$(dirname "${ENV_PREFIX}")"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  CONDA_PKGS_DIRS="${CONDA_CACHE}" mamba create -y -p "${ENV_PREFIX}" python=3.8.20 pip=24.3.1
fi

"${ENV_PREFIX}/bin/python" -m pip install --requirement "${PROJECT_ROOT}/requirements-icassp.txt"

echo
echo "Environment ready: ${ENV_PREFIX}"
echo "Activate with: conda activate ${ENV_PREFIX}"
