#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (A40/A100, PyTorch template). Clone the repo, then:
#   bash scripts/runpod_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade pip
# PyTorch images ship some packages (e.g. blinker) installed by Debian, without a pip RECORD,
# so pip cannot uninstall them to upgrade them. Install over them.
python -m pip install --ignore-installed blinker
# The pod template already ships a CUDA build of torch, so leave it alone.
grep -v '^torch' requirements.txt | python -m pip install -r /dev/stdin
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
