import boto3
import numpy as np
import pandas as pd
import cv2
import os
import json

import os
s3 = boto3.client('s3', aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
bucket = 'daml-multimodal-ncde'

os.makedirs('adni_data_v2/optical_flow', exist_ok=True)
os.makedirs('adni_data_v2/yolo/train/Normal', exist_ok=True)
os.makedirs('adni_data_v2/yolo/train/MCI', exist_ok=True)
os.makedirs('adni_data_v2/yolo/train/Severe', exist_ok=True)
os.makedirs('adni_data_v2/yolo/val/Normal', exist_ok=True)
os.makedirs('adni_data_v2/yolo/val/MCI', exist_ok=True)
os.makedirs('adni_data_v2/yolo/val/Severe', exist_ok=True)

print("Fetching S3 task2_manifest...")
s3.download_file(bucket, 'adni/week3/task2_manifest.csv', 'adni_data_v2/manifest.csv')
s3.download_file(bucket, 'adni/week3/task2_splits.json', 'adni_data_v2/splits.json')
df = pd.read_csv('adni_data_v2/manifest.csv')
with open('adni_data_v2/splits.json', 'r') as f:
    splits = json.load(f)

# Collect all volume files in the cache (paginated)
print("Listing S3 volumes (Paginated)...")
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket, Prefix='adni/week3/task2_volume_cache/')
volumes = []
for page in pages:
    for obj in page.get('Contents', []):
        if obj['Key'].endswith('.npy'):
            volumes.append(obj['Key'])

print(f"Found {len(volumes)} total cached volumes in S3.")
volumes.sort()

grouped = df.groupby('RID')
patient_sequences = {}
vol_idx = 0

for rid, group in grouped:
    seq_len = len(group)
    if vol_idx + seq_len <= len(volumes):
        patient_sequences[int(rid)] = {
            'volumes': volumes[vol_idx:vol_idx+seq_len],
            'adas13': group['ADAS13'].values,
            'times': group['MONTH_SINCE_BASELINE'].values
        }
        vol_idx += seq_len
    else:
        break

print(f"Mapped {len(patient_sequences)} patients encompassing {vol_idx} scans.")

all_flows = []
all_targets = []
all_times = []
all_rids = []

for rid, data in patient_sequences.items():
    print(f"Processing RID {rid}...")
    slices = []
    for v in data['volumes']:
        local_v = f"adni_data_v2/temp_{os.path.basename(v)}"
        if not os.path.exists(local_v):
            s3.download_file(bucket, v, local_v)
        vol = np.load(local_v)
        os.remove(local_v)
        slices.append(vol[vol.shape[0]//2])
    
    # Compute Farneback
    for i in range(len(slices) - 1):
        prev = cv2.normalize(slices[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        nxt = cv2.normalize(slices[i+1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        prev = cv2.resize(prev, (64, 64))
        nxt = cv2.resize(nxt, (64, 64))
        
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        all_flows.append(flow)
        
        target = data['adas13'][i]
        # Clean target if it is NaN
        if np.isnan(target):
            target = 10.0
        all_targets.append(target)
        all_times.append(data['times'][i])
        all_rids.append(rid)
        
        split = 'train' if (rid in splits.get('train_rids', [])) else 'val'
        severity = 'Normal'
        if target >= 20: severity = 'Severe'
        elif target >= 10: severity = 'MCI'
        
        hsv = np.zeros((64, 64, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(f"adni_data_v2/yolo/{split}/{severity}/flow_{rid}_{i}.jpg", rgb)

np.save('adni_data_v2/flows.npy', np.array(all_flows))
np.save('adni_data_v2/targets.npy', np.array(all_targets))
np.save('adni_data_v2/times.npy', np.array(all_times))
np.save('adni_data_v2/rids.npy', np.array(all_rids))
print("Data Preparation V2 complete!")
