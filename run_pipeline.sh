#!/bin/bash
set -e

echo "Setting up virtual environment..."
python3 -m venv myenv
source myenv/bin/activate

echo "Installing requirements..."
pip install torch torchvision numpy pandas boto3 opencv-python-headless ultralytics torchcde matplotlib setuptools

echo "=== Running Data Preparation ==="
python3 data_prep.py

echo "=== Running Flow Matching Model ==="
python3 flow_matching_model.py

echo "=== Running Ultralytics YOLO Model ==="
python3 yolo_model.py

echo "=== Running Neural CDE Validation ==="
python3 neural_cde_eval.py

echo "Pipeline complete! Performance graphs saved."
