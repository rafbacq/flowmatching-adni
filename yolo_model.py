from ultralytics import YOLO
import numpy as np
import os
import shutil

print("Setting up YOLO format for optical flow frames...")
for idx in range(len(os.listdir('adni_data/yolo_images/train'))):
    fname = f'adni_data/yolo_images/train/flow_{idx}.jpg'
    if os.path.exists(fname):
        for split in ['train', 'val']:
            cls_dir = f'adni_data/yolo_images_cls/{split}/target'
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copy(fname, f'{cls_dir}/flow_{idx}.jpg')

print("Training YOLO Model...")
model = YOLO('yolov8n-cls.pt')
# For YOLO classification, passing the root directory of the train/val structure is sufficient
results = model.train(data='adni_data/yolo_images_cls', epochs=3, imgsz=64, device='cpu')

embeds = []
targets = np.load('adni_data/targets.npy')
print("Extracting YOLO embeddings for Neural CDE...")
for i in range(len(targets)):
    fname = f'adni_data/yolo_images_cls/train/target/flow_{i}.jpg'
    res = model(fname)
    probs = res[0].probs.data.cpu().numpy()
    if len(probs) < 32:
        probs = np.pad(probs, (0, 32 - len(probs)))
    embeds.append(probs[:32])

np.save('adni_data/yolo_embeddings.npy', np.array(embeds))
print(f"YOLO embeddings saved. Shape: {np.array(embeds).shape}")
