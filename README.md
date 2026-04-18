# ADNI Flow Matching, YOLO, and Neural CDE Predictive Pipeline

This repository contains a high-performance machine learning pipeline on AWS EC2 for ADNI multimodal data analysis. The pipeline leverages authentic S3 data alignments to train robust Optimal Transport Flow Matching, multi-class YOLO classification, and longitudinal Neural CDE trajectory models.

## Architecture Pipeline
- **Data Preparation (`data_prep_v3.py`)**: Robust data mapping securely recovering caching volumes from AWS S3, reversing SHA1 hashes, and aligning with clinical trajectory timelines.
- **OT Flow Matching (`ot_flow_matching.py`)**: Generative models predicting exact biological optical flow targets from standard Gaussian noise.
- **Multi-Class YOLO severity classification (`yolo_advanced.py`)**: Trained classification model categorizing the severity from Normal to Severe mapped via ADAS13. Includes Ultralytics 32-D bottleneck feature extraction.
- **Batched Neural CDE (`cde_advanced.py`)**: A time-aware trajectory estimation for highly variable MRI sequences, modeling temporal cognitive scores dynamically.

## Usage
Run the entire ML flow reliably:
```
export AWS_ACCESS_KEY_ID=YOUR_ROOT_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
export FM_EPOCHS=150
export YOLO_EPOCHS=40
export CDE_EPOCHS=250
./run.sh
```

## Results
The `results/` folder stores evaluated confusion matrices, regression trajectories (predicted vs pure true targets), and flow field generative demonstrations generated straight from runtime on A10G instances.