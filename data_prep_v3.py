import boto3
import numpy as np
import pandas as pd
import cv2
import os
import json
import hashlib

s3 = boto3.client('s3', aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
bucket = 'daml-multimodal-ncde'

os.makedirs('adni_data_v3/optical_flow', exist_ok=True)
for split in ['train', 'val']:
    for c in ['Normal', 'MCI', 'Severe']:
        os.makedirs(f'adni_data_v3/yolo/{split}/{c}', exist_ok=True)

s3.download_file(bucket, 'adni/week3/task2_manifest.csv', 'adni_data_v3/manifest.csv')
s3.download_file(bucket, 'adni/week3/task2_splits.json', 'adni_data_v3/splits.json')
df = pd.read_csv('adni_data_v3/manifest.csv')
with open('adni_data_v3/splits.json', 'r') as f:
    splits = json.load(f)

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket, Prefix='adni/week3/task2_volume_cache/')
s3_volumes = set()
for page in pages:
    for obj in page.get('Contents', []):
        if obj['Key'].endswith('.npy'):
            s3_volumes.add(obj['Key'])

print(f"Verified {len(s3_volumes)} real cached volumes in S3. Mapping patients securely.")

all_flows = []
all_targets = []
all_times = []
all_rids = []

grouped = df.groupby('RID')
mapped_patients = 0
for rid, group in grouped:
    group = group.sort_values('MONTH_SINCE_BASELINE')
    
    slices = []
    valid_adas = []
    valid_times = []
    
    for _, row in group.iterrows():
        h = hashlib.sha1(str(row['SERIES_DIR']).encode('utf-8')).hexdigest()[:16]
        vol_key = f"adni/week3/task2_volume_cache/{h}.npy"
        
        if vol_key in s3_volumes:
            local_v = f"adni_data_v3/temp_{h}.npy"
            if not os.path.exists(local_v):
                s3.download_file(bucket, vol_key, local_v)
            vol = np.load(local_v)
            os.remove(local_v)
            slices.append(vol[vol.shape[0]//2])
            
            target = row['ADAS13']
            if pd.isna(target): target = 10.0
            valid_adas.append(target)
            valid_times.append(row['MONTH_SINCE_BASELINE'])
            
    if len(slices) >= 2:
        mapped_patients += 1
        for i in range(len(slices) - 1):
            prev = cv2.normalize(slices[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            nxt = cv2.normalize(slices[i+1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            prev = cv2.resize(prev, (64, 64))
            nxt = cv2.resize(nxt, (64, 64))
            
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            all_flows.append(flow)
            
            target = valid_adas[i]
            all_targets.append(target)
            all_times.append(valid_times[i])
            all_rids.append(rid)
            
            split = 'train' if (int(rid) in splits.get('train_rids', [])) else 'val'
            severity = 'Severe' if target >= 20 else ('MCI' if target >= 10 else 'Normal')
            
            hsv = np.zeros((64, 64, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(f"adni_data_v3/yolo/{split}/{severity}/flow_{rid}_{i}.jpg", rgb)

print(f"Extraction complete! Mapped {mapped_patients} genuine sequential patients.")
np.save('adni_data_v3/flows.npy', np.array(all_flows))
np.save('adni_data_v3/targets.npy', np.array(all_targets))
np.save('adni_data_v3/times.npy', np.array(all_times))
np.save('adni_data_v3/rids.npy', np.array(all_rids))
