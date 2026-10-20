#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (A40/A100, PyTorch template). Clone the repo, then:
#   bash scripts/runpod_setup.sh
set -uo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade pip

# The pod template already ships a CUDA build of torch, so leave it alone.
grep -v '^torch' requirements.txt > /tmp/reqs.txt

# PyTorch images ship some Python packages installed by Debian (blinker, cryptography, ...).
# They have no pip RECORD file, so pip cannot uninstall them to upgrade them. When that
# happens, install the offending package over the Debian copy and retry.
for attempt in $(seq 1 15); do
  if python -m pip install -r /tmp/reqs.txt 2>&1 | tee /tmp/pip.log; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    break
  fi
  pkg=$(grep -oP 'Cannot uninstall \K[^ ]+' /tmp/pip.log | head -1)
  if [ -z "$pkg" ]; then
    echo "pip failed for a different reason; see the output above"; exit 1
  fi
  echo ">>> $pkg is Debian-managed; installing over it (attempt $attempt)"
  python -m pip install --ignore-installed "$pkg" || exit 1
done

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
