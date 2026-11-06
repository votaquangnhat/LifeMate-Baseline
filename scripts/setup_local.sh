#!/usr/bin/env bash
# Run from Ubuntu/WSL. Safe to rerun: never recreates an existing environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'uv is missing. Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
if [[ ! -x .venv-local/bin/python ]]; then
    uv venv --python 3.12 .venv-local
fi
uv pip install --python .venv-local/bin/python 'torch==2.6.0+cu124' --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv-local/bin/python -r requirements.txt
.venv-local/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable in WSL'; print('GPU:', torch.cuda.get_device_name(0))"
.venv-local/bin/python scripts/download_models.py --profile smoke
.venv-local/bin/python -m baseline.doctor --config configs/local.json
echo 'Setup complete. Next: bash scripts/run_local.sh'
