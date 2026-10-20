#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: bash sh/shS/test_mask_guided_physio_behavior_modular_direct384_temporalmotion_windowpool_histproto.sh GPU_ID CHECKPOINT" >&2
  exit 2
fi
CHECKPOINT="$(realpath "$2")"
cd "$(dirname "$0")/.."
python ../src/main/train.py --mode test --gpuid "$1" --conf_file confS/dino/Test_dinov3_clip_mask_guided_physio_behavior_modular_direct384_temporalmotion_windowpool_histproto.yaml --checkpoint "$CHECKPOINT"
