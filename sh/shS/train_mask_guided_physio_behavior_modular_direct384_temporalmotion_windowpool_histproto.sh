#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python ../src/main/train.py --mode train --gpuid "${1:-0}" --conf_file confS/dino/Test_dinov3_clip_mask_guided_physio_behavior_modular_direct384_temporalmotion_windowpool_histproto.yaml
