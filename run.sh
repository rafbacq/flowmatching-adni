#!/usr/bin/env bash
# End-to-end ADNI pipeline on a CUDA-capable Ubuntu EC2 instance.
# Expects AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in the environment.
#
# Override any hyperparam via env, e.g.:
#   FM_EPOCHS=120 YOLO_EPOCHS=40 CDE_EPOCHS=200 ./run.sh

set -euo pipefail

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  echo "ERROR: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY before running." >&2
  exit 1
fi

echo "==> venv + dependencies"
if [[ ! -d myenv ]]; then
  python3 -m venv myenv
fi
# shellcheck disable=SC1091
source myenv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

mkdir -p results

echo "==> 1/5 data prep (S3 + Farneback optical flow)"
python3 data_prep_v3.py

echo "==> 2/5 OT Flow Matching"
python3 ot_flow_matching.py

echo "==> 3/5 Ultralytics YOLO classifier"
python3 yolo_advanced.py

echo "==> 4/5 Neural CDE longitudinal model"
python3 cde_advanced.py

echo "==> 5/5 Unified evaluation + comparison plots"
python3 evaluate.py

echo
echo "Pipeline finished. Figures are in ./results/"
ls -la results/
