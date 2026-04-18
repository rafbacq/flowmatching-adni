# flowmatching-adni

Alzheimer's disease progression (ADAS13) from ADNI MRI volumes, via a
three-stage pipeline:

1. **Optical-flow preprocessing** – Farneback flow between consecutive
   visits (`data_prep_v3.py`).
2. **OT Flow Matching** over flow images to get a 64-dim dynamical
   embedding per visit (`ot_flow_matching.py`).
3. **Ultralytics YOLOv8-cls** on the RGB-encoded flow images; backbone
   features (not class probabilities) are exported as a second embedding
   (`yolo_advanced.py`).
4. **Neural CDE** on the per-patient longitudinal sequence of
   `[time ‖ FM_embed ‖ YOLO_embed]` to predict ADAS13 over time
   (`cde_advanced.py`).
5. **evaluate.py** reports MAE / RMSE / R² against ridge-regression and
   mean baselines and writes the comparison figures.

## Running on an EC2 instance

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
./run.sh
```

Hyperparameters can be overridden via env vars:

```bash
FM_EPOCHS=120 YOLO_EPOCHS=40 CDE_EPOCHS=200 ./run.sh
```

## Outputs

Everything lands under `results/`:

| File | What it shows |
| --- | --- |
| `fm_training_loss.png` | OT Flow-Matching velocity-MSE per epoch |
| `fm_samples.png` | real vs. model-sampled flow magnitudes |
| `yolo_confusion.png` | YOLO 3-class confusion matrix (val) |
| `cde_training_curves.png` | train/val MSE and val MAE per epoch |
| `cde_trajectories.png` | predicted vs. true ADAS13 for val patients |
| `cde_scatter.png` | predicted-vs-true scatter (val) |
| `cde_error_hist.png` | residual histogram (val) |
| `model_comparison.png` | RMSE bars: mean / FM-ridge / YOLO-ridge / FM+YOLO-ridge / Neural CDE |
| `summary.png` | two-panel scatter: best ridge probe vs Neural CDE |
| `summary.json` | machine-readable metrics for all models |

## Data layout (produced by `data_prep_v3.py`)

```
adni_data_v3/
  manifest.csv            # task2 manifest from S3
  splits.json             # train/val RIDs
  target_stats.json       # z-score stats (train only)
  flows.npy               # (N, H, W, 2) Farneback flows
  targets.npy             # (N,) ADAS13 at visit i
  times.npy               # (N,) months-since-baseline
  rids.npy                # (N,) patient IDs
  visit_ix.npy            # (N,) 0-based visit index within patient
  fm_embeddings.npy       # (N, 64)  written by ot_flow_matching.py
  yolo_embeddings.npy     # (N, D)   written by yolo_advanced.py
  yolo/{train,val}/{Normal,MCI,Severe}/*.jpg
  cache/                  # downloaded .npy volumes (reused across runs)
```

## Key design points

- **Patient ↔ volume mapping uses `SHA1(SERIES_DIR)[:16]`** – the same
  hash the cache in S3 is keyed on. Every flow pair is tied to the
  correct RID, visit, and ADAS13 label.
- **Train/val split is honoured end-to-end.** Target z-score statistics,
  feature standardisation, and ridge probes are fit on train only.
- **YOLO embeddings are backbone features**, extracted via a forward
  hook on the penultimate layer (the 3-class probabilities from the old
  version carry almost no information).
- **Neural CDE vector field is an MLP with tanh nonlinearity** and the
  model trains on batches of padded patient sequences, not one patient
  at a time.
