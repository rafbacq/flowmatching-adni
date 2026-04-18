#!/bin/bash
set -e
echo "Starting Advanced ADNI Flow Machine Pipeline"
source myenv/bin/activate
pip install numpy pandas boto3 opencv-python-headless ultralytics torchcde matplotlib
python3 data_prep_advanced.py
python3 ot_flow_matching.py
python3 yolo_advanced.py
python3 cde_advanced.py
echo "ALL ADVANCED PIPELINE SCRIPTS FINISHED!"
