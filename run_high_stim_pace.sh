#!/bin/bash
# Run ON PACE, on a GPU node, AFTER:
#   module load anaconda3/2022.05 && source ~/.bashrc && conda activate tribe
#
# Upload first (from your Mac):
#   scp high_stim.mp4 vbhatt35@login-phoenix-gnr-1.pace.gatech.edu:~/tribe-test/

set -euo pipefail
# Large caches must not use small home quota (tribe_v2.py also sets these when unset on Phoenix scratch)
export UV_CACHE_DIR="${UV_CACHE_DIR:-/storage/scratch1/1/${USER}/.uv-cache}"
export HF_HOME="${HF_HOME:-/storage/scratch1/1/${USER}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

cd ~/tribe-test

python tribe_v2.py \
  --backend nforge \
  --video ~/tribe-test/high_stim.mp4 \
  --output ~/tribe-test/high_stim_preds.npz \
  --cache-dir ~/tribe-test/cache_high_stim

echo "Done. Download (from Mac): scp vbhatt35@login-phoenix-gnr-1.pace.gatech.edu:~/tribe-test/high_stim_preds.npz ."
