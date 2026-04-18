import boto3
import numpy as np
import pandas as pd
import cv2
import os

import os
s3 = boto3.client('s3', aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
bucket = 'daml-multimodal-ncde'

os.makedirs('adni_data/yolo_images/train', exist_ok=True)

print("Downloading manifest...")
s3.download_file(bucket, 'adni/week3/task2_manifest.csv', 'adni_data/task2_manifest.csv')
df = pd.read_csv('adni_data/task2_manifest.csv')

response = s3.list_objects_v2(Bucket=bucket, Prefix='adni/week3/task2_volume_cache/')
volumes = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.npy')][:20]

print(f"Downloading {len(volumes)} MRI volumes for optical flow extraction...")
slices = []
for v in volumes:
    local_path = f"adni_data/{os.path.basename(v)}"
    s3.download_file(bucket, v, local_path)
    vol = np.load(local_path)
    mid_idx = vol.shape[0] // 2
    slice_2d = vol[mid_idx]
    slice_2d = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255).astype(np.uint8)
    slices.append(slice_2d)

print("Computing continuous optical flow trajectories...")
flows = []
targets = []
times = []
for i in range(len(slices) - 1):
    prev, nxt = slices[i], slices[i+1]
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Downsample for faster computation in models later
    flow = cv2.resize(flow, (64, 64))
    flows.append(flow)
    
    hsv = np.zeros((64, 64, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'adni_data/yolo_images/train/flow_{i}.jpg', rgb)
    
    targets.append(df['ADAS13'].iloc[i])
    times.append(df['MONTH_SINCE_BASELINE'].iloc[i])

np.save('adni_data/flows.npy', np.array(flows))
np.save('adni_data/targets.npy', np.array(targets))
np.save('adni_data/times.npy', np.array(times))
print("Data prep complete.")
