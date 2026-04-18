#!/bin/bash
set -e
echo "Starting V3 Genuine ADNI Machine Learning Pipeline"
source myenv/bin/activate
pip install numpy pandas boto3 opencv-python-headless ultralytics torchcde matplotlib xxhash

echo "=== Data Preparation V3 ==="
python3 data_prep_v3.py

# Replace directories in subsequent models automatically via sed on the EC2 natively
sed -i 's/adni_data_v2/adni_data_v3/g' ot_flow_matching.py
sed -i 's/adni_data_v2/adni_data_v3/g' yolo_advanced.py
sed -i 's/adni_data_v2/adni_data_v3/g' cde_advanced.py

echo "=== Flow Matching V3 ==="
python3 ot_flow_matching.py

echo "=== Ultralytics YOLO V3 ==="
python3 yolo_advanced.py

echo "=== Neural CDE V3 ==="
python3 cde_advanced.py

echo "GENUINE PIPELINE EVALUATIONS COMPLETE!"
