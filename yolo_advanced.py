"""
yolo_advanced.py – Multi-class YOLOv8-cls severity classifier.
Reads optical-flow images from DATA/yolo/{train,val}/{Normal,MCI,Severe}/.
Outputs:
  • yolo_embeddings.npy   – per-sample probability vectors (padded to 32-d)
  • yolo_confusion.png    – Ultralytics confusion matrix
  • yolo_metrics.png      – top-1 / top-5 accuracy curves
"""

from ultralytics import YOLO
import numpy as np, os, shutil, json, glob, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.environ.get("DATA_DIR", "adni_data")
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "15"))

# ── load metadata ────────────────────────────────────────────────────────
with open(f"{DATA}/splits.json") as f:
    splits = json.load(f)
rids = np.load(f"{DATA}/rids.npy")
targets = np.load(f"{DATA}/targets.npy")

# ── ensure val is non-empty (move 20 % of train if needed) ──────────────
train_imgs = glob.glob(f"{DATA}/yolo/train/*/*.jpg")
val_imgs = glob.glob(f"{DATA}/yolo/val/*/*.jpg")
if len(val_imgs) == 0 and len(train_imgs) > 0:
    random.seed(42)
    random.shuffle(train_imgs)
    for f in train_imgs[: max(1, len(train_imgs) // 5)]:
        dst = f.replace("/train/", "/val/")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(f, dst)
    print(f"[YOLO] Moved {max(1, len(train_imgs)//5)} images train→val")

# ── train ────────────────────────────────────────────────────────────────
print(f"[YOLO] Training YOLOv8n-cls for {EPOCHS} epochs …")
model = YOLO("yolov8n-cls.pt")
results = model.train(
    data=f"{DATA}/yolo",
    epochs=EPOCHS,
    imgsz=64,
    batch=16,
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    verbose=True,
)

# ── copy confusion matrix produced by Ultralytics ────────────────────────
for candidate in [
    f"{results.save_dir}/confusion_matrix.png",
    f"{results.save_dir}/confusion_matrix_normalized.png",
]:
    if os.path.exists(candidate):
        shutil.copy(candidate, f"{DATA}/yolo_confusion.png")
        print(f"[YOLO] Confusion matrix → {DATA}/yolo_confusion.png")
        break

# ── custom accuracy-curve plot from Ultralytics CSV ──────────────────────
csv_path = f"{results.save_dir}/results.csv"
if os.path.exists(csv_path):
    import pandas as pd
    res_df = pd.read_csv(csv_path)
    res_df.columns = [c.strip() for c in res_df.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in res_df.columns:
        if "acc" in col.lower() or "loss" in col.lower():
            ax.plot(res_df[col].values, label=col, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.set_title("YOLO Classification – Training Metrics")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{DATA}/yolo_metrics.png", dpi=150); plt.close()
    print(f"[YOLO] Metrics curve → {DATA}/yolo_metrics.png")

# ── extract embeddings (probability vectors) per sample ──────────────────
embeds = []
local_idx: dict[int, int] = {}
print("[YOLO] Extracting embeddings …")
for i in range(len(rids)):
    rid = int(rids[i])
    target = float(targets[i])
    img_i = local_idx.get(rid, 0)
    local_idx[rid] = img_i + 1

    sev = "Severe" if target >= 20 else ("MCI" if target >= 10 else "Normal")
    for sp in ("train", "val"):
        path = f"{DATA}/yolo/{sp}/{sev}/flow_{rid}_{img_i}.jpg"
        if os.path.exists(path):
            break
    else:
        print(f"  ⚠ missing image for RID={rid} idx={img_i}, skipping")
        embeds.append(np.zeros(32))
        continue

    res = model(path, verbose=False)
    probs = res[0].probs.data.cpu().numpy()
    if len(probs) < 32:
        probs = np.pad(probs, (0, 32 - len(probs)))
    embeds.append(probs[:32])

np.save(f"{DATA}/yolo_embeddings.npy", np.array(embeds))
print(f"[YOLO] Saved embeddings → {DATA}/yolo_embeddings.npy  shape={np.array(embeds).shape}")
print("[YOLO] Done.")
