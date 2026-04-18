"""
Ultralytics YOLO (classification) on optical-flow RGB images.

Three classes: Normal / MCI / Severe, derived from ADAS13 thresholds.

Replaces the old version which used the 3-dim class-probability vector
(padded with zeros to 32) as the "embedding" -- information-free. Instead we
register a forward hook on the penultimate layer (the pooled backbone feature)
and export that as a dense embedding for the Neural CDE.

Outputs:
  - trained YOLO weights (ultralytics default location)
  - adni_data_v3/yolo_embeddings.npy   (N, D)  per-sample backbone feature
  - results/yolo_confusion.png
  - results/yolo_metrics.json
"""

import os
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

DATA_DIR = "adni_data_v3"
RESULTS_DIR = "results"
IMGSZ = 128
EPOCHS = int(os.environ.get("YOLO_EPOCHS", 25))
BATCH = int(os.environ.get("YOLO_BATCH", 32))


def build_ordered_paths():
    """Return, for every flow-pair i in the saved arrays, the image path on
    disk (either train/ or val/ depending on the split)."""
    rids = np.load(f"{DATA_DIR}/rids.npy")
    targets = np.load(f"{DATA_DIR}/targets.npy")
    visit_ix = np.load(f"{DATA_DIR}/visit_ix.npy")
    with open(f"{DATA_DIR}/splits.json") as f:
        splits = json.load(f)
    train_rids = set(int(x) for x in splits.get("train_rids", []))

    def sev(a):
        if a >= 20: return "Severe"
        if a >= 10: return "MCI"
        return "Normal"

    paths = []
    for rid, t, vi in zip(rids, targets, visit_ix):
        split = "train" if int(rid) in train_rids else "val"
        p = f"{DATA_DIR}/yolo/{split}/{sev(t)}/flow_{int(rid)}_{int(vi)}.jpg"
        paths.append(p)
    return paths


def attach_feature_hook(model):
    """Register a hook on the input of the final Classify head so we capture
    its pre-logit (backbone) features. Returns (handle, getter)."""
    inner = model.model.model  # torch.nn.Sequential of layers
    target = inner[-1]  # the Classify head

    buf = {}
    def hook(module, inp, out):
        # inp is a tuple; for Classify in ultralytics it's (x,) with x already pooled+flattened.
        feat = inp[0]
        if feat.dim() > 2:
            feat = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feat, 1), 1)
        buf["f"] = feat.detach()

    handle = target.register_forward_hook(hook)
    return handle, buf


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {device}")

    n_train = len(glob.glob(f"{DATA_DIR}/yolo/train/*/*.jpg"))
    n_val = len(glob.glob(f"{DATA_DIR}/yolo/val/*/*.jpg"))
    print(f"YOLO dataset: train={n_train} val={n_val}")
    if n_train == 0:
        raise RuntimeError("No training images found. Run data_prep_v3.py first.")
    if n_val == 0:
        raise RuntimeError("No val images; check your splits.json contains train_rids.")

    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=f"{DATA_DIR}/yolo",
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        patience=10,
        project=f"{RESULTS_DIR}/yolo_runs",
        name="adni_cls",
        exist_ok=True,
        verbose=False,
    )

    # Copy the confusion matrix produced by ultralytics to a stable path.
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        src = os.path.join(str(save_dir), "confusion_matrix.png")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, f"{RESULTS_DIR}/yolo_confusion.png")

    # -------------------- Extract backbone embeddings --------------------
    model.model.to(device).eval()
    handle, buf = attach_feature_hook(model)

    paths = build_ordered_paths()
    embeds = []
    batch = []
    for p in tqdm(paths, desc="YOLO features"):
        batch.append(p)
        if len(batch) >= BATCH:
            _ = model.predict(batch, imgsz=IMGSZ, device=device, verbose=False)
            embeds.append(buf["f"].cpu().numpy())
            batch = []
    if batch:
        _ = model.predict(batch, imgsz=IMGSZ, device=device, verbose=False)
        embeds.append(buf["f"].cpu().numpy())
    handle.remove()

    embeds = np.concatenate(embeds, axis=0).astype(np.float32)
    assert embeds.shape[0] == len(paths), (embeds.shape, len(paths))
    np.save(f"{DATA_DIR}/yolo_embeddings.npy", embeds)
    print(f"Saved YOLO embeddings: {embeds.shape}")

    # -------------------- Summary metrics --------------------
    metrics = {}
    try:
        val_res = model.val(data=f"{DATA_DIR}/yolo", imgsz=IMGSZ, device=device, verbose=False)
        metrics["top1"] = float(getattr(val_res, "top1", float("nan")))
        metrics["top5"] = float(getattr(val_res, "top5", float("nan")))
    except Exception as e:
        metrics["val_error"] = str(e)
    metrics["embed_dim"] = int(embeds.shape[1])
    metrics["n_train"] = n_train
    metrics["n_val"] = n_val
    metrics["epochs"] = EPOCHS
    with open(f"{RESULTS_DIR}/yolo_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("YOLO metrics:", metrics)


if __name__ == "__main__":
    main()
