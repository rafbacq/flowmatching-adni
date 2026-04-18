#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# run.sh – End-to-end ADNI flow-matching pipeline
#
# Usage:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   ./run.sh                          # default epochs
#   FM_EPOCHS=150 YOLO_EPOCHS=40 CDE_EPOCHS=250 ./run.sh  # push harder
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

export DATA_DIR="${DATA_DIR:-adni_data}"
export FM_EPOCHS="${FM_EPOCHS:-50}"
export YOLO_EPOCHS="${YOLO_EPOCHS:-15}"
export CDE_EPOCHS="${CDE_EPOCHS:-80}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ADNI Flow Matching Pipeline"
echo "  DATA_DIR   = $DATA_DIR"
echo "  FM_EPOCHS  = $FM_EPOCHS"
echo "  YOLO_EPOCHS= $YOLO_EPOCHS"
echo "  CDE_EPOCHS = $CDE_EPOCHS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate venv if present
if [ -f myenv/bin/activate ]; then
    source myenv/bin/activate
fi

pip -q install numpy pandas boto3 opencv-python-headless ultralytics torchcde matplotlib

echo ""
echo "▶ Step 1/4  Data Preparation"
python3 data_prep_v3.py

echo ""
echo "▶ Step 2/4  OT Flow Matching"
python3 ot_flow_matching.py

echo ""
echo "▶ Step 3/4  YOLO Classification"
python3 yolo_advanced.py

echo ""
echo "▶ Step 4/4  Neural CDE"
python3 cde_advanced.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Pipeline complete!"
echo "  Graphs in $DATA_DIR/:"
ls -1 "$DATA_DIR"/*.png 2>/dev/null || echo "  (no graphs found)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
