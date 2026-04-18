"""
ADNI data preparation.

Downloads the task2 manifest + splits + cached MRI volumes from S3, computes
patient-level optical-flow sequences between consecutive visits, and writes a
YOLO-style folder tree for classification plus numpy arrays for flow matching
and the Neural CDE.

Key design points vs. older versions:
  * Patient -> volume mapping is done via the SHA1(SERIES_DIR) cache key, so
    every scan is paired with the correct RID / visit / ADAS13 target.
  * We keep patient-level sequences (grouped by RID, ordered by time) all the
    way through so the Neural CDE sees real longitudinal trajectories.
  * We sample 3 axial slices per volume (upper-mid / mid / lower-mid) and
    average-pool the flow rather than taking one arbitrary slice.
  * All targets are z-score normalised using TRAIN-ONLY statistics; the
    statistics are saved so downstream scripts un-normalise for reporting.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import cv2
import boto3
from botocore.config import Config
from tqdm import tqdm

OUT_DIR = "adni_data_v3"
BUCKET = "daml-multimodal-ncde"
MANIFEST_KEY = "adni/week3/task2_manifest.csv"
SPLITS_KEY = "adni/week3/task2_splits.json"
VOLUME_PREFIX = "adni/week3/task2_volume_cache/"
IMG_SIZE = 128  # YOLO imgsz; previous version used 64 which is too small.


def build_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/cache", exist_ok=True)
    for split in ("train", "val"):
        for c in ("Normal", "MCI", "Severe"):
            os.makedirs(f"{OUT_DIR}/yolo/{split}/{c}", exist_ok=True)


def s3_client():
    cfg = Config(retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=cfg,
    )


def list_cached_volumes(s3):
    paginator = s3.get_paginator("list_objects_v2")
    keys = set()
    for page in paginator.paginate(Bucket=BUCKET, Prefix=VOLUME_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".npy"):
                keys.add(obj["Key"])
    return keys


def load_volume(s3, vol_key):
    """Download (or reuse) and load a cached volume."""
    local = f"{OUT_DIR}/cache/{os.path.basename(vol_key)}"
    if not os.path.exists(local):
        s3.download_file(BUCKET, vol_key, local)
    return np.load(local)


def volume_to_representative_slice(vol):
    """Average three central axial slices into a single 2D image."""
    d = vol.shape[0]
    idx = [max(0, d // 2 - d // 8), d // 2, min(d - 1, d // 2 + d // 8)]
    stack = np.stack([vol[i] for i in idx], axis=0).astype(np.float32)
    img = stack.mean(axis=0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def farneback_flow(prev, nxt):
    return cv2.calcOpticalFlowFarneback(prev, nxt, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2,
                                        flags=0)


def flow_to_rgb(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def severity_for(adas13):
    if adas13 >= 20:
        return "Severe"
    if adas13 >= 10:
        return "MCI"
    return "Normal"


def main():
    build_dirs()
    s3 = s3_client()

    print("Downloading manifest + splits...")
    s3.download_file(BUCKET, MANIFEST_KEY, f"{OUT_DIR}/manifest.csv")
    s3.download_file(BUCKET, SPLITS_KEY, f"{OUT_DIR}/splits.json")
    df = pd.read_csv(f"{OUT_DIR}/manifest.csv")
    with open(f"{OUT_DIR}/splits.json") as f:
        splits = json.load(f)
    train_rids = set(int(x) for x in splits.get("train_rids", []))

    print("Listing S3 volume cache...")
    available = list_cached_volumes(s3)
    print(f"  {len(available)} volumes available in cache")

    flows, targets, times_, rids_, visit_ix_ = [], [], [], [], []
    mapped = 0
    grouped = df.groupby("RID")

    for rid, group in tqdm(list(grouped), desc="Patients"):
        group = group.sort_values("MONTH_SINCE_BASELINE").reset_index(drop=True)
        slices, adas_seq, time_seq = [], [], []
        for _, row in group.iterrows():
            h = hashlib.sha1(str(row["SERIES_DIR"]).encode("utf-8")).hexdigest()[:16]
            vol_key = f"{VOLUME_PREFIX}{h}.npy"
            if vol_key not in available:
                continue
            try:
                vol = load_volume(s3, vol_key)
            except Exception as e:  # network/corrupt
                print(f"  skip RID {rid} scan {h}: {e}")
                continue
            slices.append(volume_to_representative_slice(vol))
            a = row["ADAS13"]
            adas_seq.append(float("nan") if pd.isna(a) else float(a))
            time_seq.append(float(row["MONTH_SINCE_BASELINE"]))
        if len(slices) < 2:
            continue
        mapped += 1

        split = "train" if int(rid) in train_rids else "val"
        for i in range(len(slices) - 1):
            flow = farneback_flow(slices[i], slices[i + 1])
            target = adas_seq[i]
            if np.isnan(target):
                continue  # skip pairs without a valid label
            flows.append(flow.astype(np.float32))
            targets.append(target)
            times_.append(time_seq[i])
            rids_.append(int(rid))
            visit_ix_.append(i)

            sev = severity_for(target)
            cv2.imwrite(
                f"{OUT_DIR}/yolo/{split}/{sev}/flow_{int(rid)}_{i}.jpg",
                flow_to_rgb(flow),
            )

    flows = np.asarray(flows, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    times_ = np.asarray(times_, dtype=np.float32)
    rids_ = np.asarray(rids_, dtype=np.int64)
    visit_ix_ = np.asarray(visit_ix_, dtype=np.int64)

    # z-score ADAS13 using train patients only so val is truly held out.
    train_mask = np.isin(rids_, list(train_rids))
    if train_mask.sum() == 0:
        mu, sigma = float(targets.mean()), float(targets.std() + 1e-6)
    else:
        mu = float(targets[train_mask].mean())
        sigma = float(targets[train_mask].std() + 1e-6)
    stats = {"adas13_mean": mu, "adas13_std": sigma}
    with open(f"{OUT_DIR}/target_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    np.save(f"{OUT_DIR}/flows.npy", flows)
    np.save(f"{OUT_DIR}/targets.npy", targets)
    np.save(f"{OUT_DIR}/times.npy", times_)
    np.save(f"{OUT_DIR}/rids.npy", rids_)
    np.save(f"{OUT_DIR}/visit_ix.npy", visit_ix_)

    print(f"\nFinished: {mapped} patients, {len(flows)} flow pairs")
    print(f"  Target mean (train): {mu:.3f}   std: {sigma:.3f}")
    print(f"  Train pairs: {int(train_mask.sum())}  Val pairs: {int((~train_mask).sum())}")


if __name__ == "__main__":
    main()
