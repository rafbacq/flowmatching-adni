from ultralytics import YOLO
import numpy as np
import os
import shutil
import json

with open('adni_data_v2/splits.json', 'r') as f:
    splits = json.load(f)

import glob
train_files = glob.glob('adni_data_v2/yolo/train/*/*.jpg')
val_files = glob.glob('adni_data_v2/yolo/val/*/*.jpg')
if len(val_files) == 0 and len(train_files) > 0:
    import random
    random.shuffle(train_files)
    move_count = max(1, len(train_files) // 5)
    for f in train_files[:move_count]:
        dest = f.replace('/train/', '/val/')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(f, dest)

print("Training Segmented Multi-Class YOLO...")
model = YOLO('yolov8n-cls.pt')

# Need to map to a dataset.yaml or simply pass the folder path
# Ultralytics classification automatically uses folder names as classes
results = model.train(data='adni_data_v2/yolo', epochs=5, imgsz=64, device='cuda')

conf_mat_path = f"{results.save_dir}/confusion_matrix.png"
if os.path.exists(conf_mat_path):
    shutil.copy(conf_mat_path, "adni_data_v2/yolo_confusion.png")

rids = np.load('adni_data_v2/rids.npy')
targets = np.load('adni_data_v2/targets.npy')

embeds = []
print("Extracting multi-class embeddings...")
local_idx = {}
for i in range(len(rids)):
    rid = int(rids[i])
    target = targets[i]
    if rid not in local_idx:
        local_idx[rid] = 0
    img_i = local_idx[rid]
    local_idx[rid] += 1
    
    split = 'train' if (rid in splits.get('train_rids', [])) else 'val'
    severity = 'Normal'
    if target >= 20: severity = 'Severe'
    elif target >= 10: severity = 'MCI'
    
    fname_train = f"adni_data_v2/yolo/train/{severity}/flow_{rid}_{img_i}.jpg"
    fname_val = f"adni_data_v2/yolo/val/{severity}/flow_{rid}_{img_i}.jpg"
    fname = fname_train if os.path.exists(fname_train) else fname_val
    res = model(fname)
    probs = res[0].probs.data.cpu().numpy()
    if len(probs) < 32:
        probs = np.pad(probs, (0, 32 - len(probs)))
    embeds.append(probs[:32])

np.save('adni_data_v2/yolo_embeddings.npy', np.array(embeds))
print("YOLO Embeddings exported!")
