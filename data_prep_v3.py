"""
data_prep_v3.py – Download ADNI volumes from S3, compute Farneback optical
flow between consecutive visits per patient, and write:
  • flows.npy          (N, 64, 64, 2)  float32
  • targets.npy        (N,)            ADAS13 score
  • times.npy          (N,)            months since baseline
  • rids.npy           (N,)            patient RID
  • yolo/{train,val}/{Normal,MCI,Severe}/*.jpg
"""

import boto3, numpy as np, pandas as pd, cv2, os, json, hashlib

DATA = os.environ.get("DATA_DIR", "adni_data")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)
BUCKET = "daml-multimodal-ncde"

# ── directories ──────────────────────────────────────────────────────────
os.makedirs(f"{DATA}/optical_flow", exist_ok=True)
for sp in ("train", "val"):
    for c in ("Normal", "MCI", "Severe"):
        os.makedirs(f"{DATA}/yolo/{sp}/{c}", exist_ok=True)

# ── manifest & splits ────────────────────────────────────────────────────
s3.download_file(BUCKET, "adni/week3/task2_manifest.csv", f"{DATA}/manifest.csv")
s3.download_file(BUCKET, "adni/week3/task2_splits.json", f"{DATA}/splits.json")

df = pd.read_csv(f"{DATA}/manifest.csv")
with open(f"{DATA}/splits.json") as f:
    splits = json.load(f)

# ── enumerate S3 volume cache ────────────────────────────────────────────
paginator = s3.get_paginator("list_objects_v2")
s3_volumes: set[str] = set()
for page in paginator.paginate(Bucket=BUCKET, Prefix="adni/week3/task2_volume_cache/"):
    for obj in page.get("Contents", []):
        if obj["Key"].endswith(".npy"):
            s3_volumes.add(obj["Key"])
print(f"[data_prep] {len(s3_volumes)} cached volumes found in S3.")

# ── per-patient longitudinal extraction ──────────────────────────────────
all_flows, all_targets, all_times, all_rids = [], [], [], []
mapped = 0

for rid, grp in df.groupby("RID"):
    grp = grp.sort_values("MONTH_SINCE_BASELINE")
    slices, adas_vals, time_vals = [], [], []

    for _, row in grp.iterrows():
        h = hashlib.sha1(str(row["SERIES_DIR"]).encode()).hexdigest()[:16]
        key = f"adni/week3/task2_volume_cache/{h}.npy"
        if key not in s3_volumes:
            continue
        local = f"{DATA}/_tmp_{h}.npy"
        try:
            s3.download_file(BUCKET, key, local)
            vol = np.load(local)
        finally:
            if os.path.exists(local):
                os.remove(local)
        slices.append(vol[vol.shape[0] // 2])
        target = row["ADAS13"]
        adas_vals.append(10.0 if pd.isna(target) else float(target))
        time_vals.append(float(row["MONTH_SINCE_BASELINE"]))

    if len(slices) < 2:
        continue
    mapped += 1

    for i in range(len(slices) - 1):
        a = cv2.resize(cv2.normalize(slices[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (64, 64))
        b = cv2.resize(cv2.normalize(slices[i + 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (64, 64))
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        all_flows.append(flow)
        all_targets.append(adas_vals[i])
        all_times.append(time_vals[i])
        all_rids.append(int(rid))

        # HSV flow visualisation → YOLO image
        split = "train" if int(rid) in splits.get("train_rids", []) else "val"
        sev = "Severe" if adas_vals[i] >= 20 else ("MCI" if adas_vals[i] >= 10 else "Normal")
        hsv = np.zeros((64, 64, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(f"{DATA}/yolo/{split}/{sev}/flow_{rid}_{i}.jpg", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

print(f"[data_prep] {mapped} patients with ≥2 visits mapped → {len(all_flows)} flow samples.")
for name, arr in [("flows", all_flows), ("targets", all_targets), ("times", all_times), ("rids", all_rids)]:
    np.save(f"{DATA}/{name}.npy", np.array(arr))
