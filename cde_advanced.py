"""
Neural CDE for longitudinal ADAS13 prediction.

Patient-level sequences are built from FM + YOLO embeddings (plus time as
channel 0 as required by torchcde). Variable-length sequences are padded with
NaN; `hermite_cubic_coefficients_with_backward_differences` handles that by
forward-filling and we later mask padded positions in the loss.

Replaces the old Neural-CDE scripts which (a) merged all patients into one
artificial sequence (`neural_cde_eval.py`) or (b) trained one patient at a
time without validation metrics and with a pure-linear vector field.

Outputs:
  - results/cde_training_curves.png
  - results/cde_trajectories.png
  - results/cde_scatter.png
  - results/cde_error_hist.png
  - results/cde_metrics.json
  - results/cde_predictions.npz  (raw predictions for evaluate.py)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = "adni_data_v3"
RESULTS_DIR = "results"

EPOCHS = int(os.environ.get("CDE_EPOCHS", 150))
HIDDEN = int(os.environ.get("CDE_HIDDEN", 48))
LR = float(os.environ.get("CDE_LR", 1e-3))
BATCH_PATIENTS = int(os.environ.get("CDE_BATCH", 8))
SEED = 0


class CDEFunc(nn.Module):
    """Vector field f_theta(z) of shape (hidden, input). MLP, not a bare Linear."""

    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, hidden_channels * input_channels),
        )

    def forward(self, t, z):
        return self.net(z).view(z.size(0), self.hidden_channels, self.input_channels).tanh()


class NeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_channels, 64), nn.ReLU(),
            nn.Linear(64, hidden_channels),
        )
        self.func = CDEFunc(input_channels, hidden_channels)
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, 32), nn.ReLU(), nn.Linear(32, output_channels)
        )

    def forward(self, coeffs, t_eval):
        X = torchcde.CubicSpline(coeffs)
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        z = torchcde.cdeint(X=X, z0=z0, func=self.func, t=t_eval, method="rk4",
                            options={"step_size": 0.5})
        return self.readout(z)


def build_patient_tensors(features, times, targets, rids):
    """Group per-pair samples into per-patient sequences, pad to max length.

    Returns:
      seqs: (B, T_max, 1 + D) with NaN padding -- channel 0 is time
      ys:   (B, T_max) float target (NaN where padded / missing)
      mask: (B, T_max) bool   valid positions
      t_eval: (T_max,) shared evaluation grid = arange(T_max)
      meta: list of per-patient dicts with raw times/targets for plotting
    """
    patients = {}
    for i, r in enumerate(rids):
        patients.setdefault(int(r), []).append(i)

    seqs, ys, masks, rids_out, metas = [], [], [], [], []
    T_max = max(len(v) for v in patients.values())
    if T_max < 2:
        raise RuntimeError("No patient has >=2 visits; can't train a CDE.")

    for rid, idxs in patients.items():
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs, key=lambda j: times[j])
        t = np.asarray([times[j] for j in idxs], dtype=np.float32)
        # Enforce strictly increasing time (torchcde requirement).
        for k in range(1, len(t)):
            if t[k] <= t[k - 1]:
                t[k] = t[k - 1] + 1e-3
        f = np.stack([features[j] for j in idxs], axis=0).astype(np.float32)
        y = np.asarray([targets[j] for j in idxs], dtype=np.float32)

        T = len(idxs)
        pad_T = T_max - T
        t_pad = np.concatenate([t, np.full(pad_T, np.nan, dtype=np.float32)])
        f_pad = np.concatenate([f, np.full((pad_T, f.shape[1]), np.nan, dtype=np.float32)], axis=0)
        y_pad = np.concatenate([y, np.full(pad_T, np.nan, dtype=np.float32)])
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(pad_T, dtype=bool)])

        # Channel 0 must be the (non-nan) time for torchcde.
        # We pass time in the first channel and use it both to define the
        # integration variable and as an input feature.
        seq = np.concatenate([t_pad[:, None], f_pad], axis=1)  # (T_max, 1+D)
        seqs.append(seq); ys.append(y_pad); masks.append(mask); rids_out.append(rid)
        metas.append({"rid": rid, "t": t, "y": y})

    seqs = torch.tensor(np.stack(seqs))        # (B, T_max, C)
    ys = torch.tensor(np.stack(ys))            # (B, T_max)
    masks = torch.tensor(np.stack(masks))      # (B, T_max)
    t_eval = torch.linspace(0.0, float(T_max - 1), T_max)
    return seqs, ys, masks, t_eval, rids_out, metas


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    fm = np.load(f"{DATA_DIR}/fm_embeddings.npy")
    yo = np.load(f"{DATA_DIR}/yolo_embeddings.npy")
    targets = np.load(f"{DATA_DIR}/targets.npy")
    times = np.load(f"{DATA_DIR}/times.npy")
    rids = np.load(f"{DATA_DIR}/rids.npy")
    with open(f"{DATA_DIR}/splits.json") as f:
        splits = json.load(f)
    with open(f"{DATA_DIR}/target_stats.json") as f:
        stats = json.load(f)
    train_rids = set(int(x) for x in splits.get("train_rids", []))
    mu, sigma = stats["adas13_mean"], stats["adas13_std"]

    features = np.concatenate([fm, yo], axis=1).astype(np.float32)
    # Standardise per-channel over the train pairs to keep CDE well-conditioned.
    train_mask_pair = np.array([int(r) in train_rids for r in rids])
    fmean = features[train_mask_pair].mean(axis=0, keepdims=True)
    fstd = features[train_mask_pair].std(axis=0, keepdims=True) + 1e-6
    features = (features - fmean) / fstd

    y_norm = (targets - mu) / sigma

    seqs, ys, masks, t_eval, rid_order, metas = build_patient_tensors(
        features, times, y_norm, rids
    )
    print(f"Patients: {seqs.shape[0]}, max visits: {seqs.shape[1]}, channels: {seqs.shape[2]}")

    is_train = torch.tensor([int(r) in train_rids for r in rid_order])
    train_idx = torch.where(is_train)[0]
    val_idx = torch.where(~is_train)[0]
    print(f"  train patients: {len(train_idx)}  val patients: {len(val_idx)}")

    in_ch = seqs.shape[2]
    model = NeuralCDE(input_channels=in_ch, hidden_channels=HIDDEN, output_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Pre-compute coeffs once (they don't depend on weights).
    coeffs_all = torchcde.hermite_cubic_coefficients_with_backward_differences(seqs).to(device)
    ys = ys.to(device); masks = masks.to(device); t_eval = t_eval.to(device)

    history = {"train_mse": [], "val_mse": [], "val_mae": []}

    for epoch in range(EPOCHS):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        epoch_losses = []
        for k in range(0, len(perm), BATCH_PATIENTS):
            batch = perm[k:k + BATCH_PATIENTS]
            c = coeffs_all[batch]
            y = ys[batch]
            m = masks[batch]
            pred = model(c, t_eval).squeeze(-1)  # (b, T_max)
            diff = (pred - y) ** 2
            loss = (diff[m]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(loss.item())
        sched.step()

        # ---- eval ----
        model.eval()
        with torch.no_grad():
            pred_val = model(coeffs_all[val_idx], t_eval).squeeze(-1)
            y_val = ys[val_idx]; m_val = masks[val_idx]
            diff = pred_val - y_val
            val_mse = (diff[m_val] ** 2).mean().item()
            val_mae = diff[m_val].abs().mean().item()
        history["train_mse"].append(float(np.mean(epoch_losses)))
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{EPOCHS}  train_mse={history['train_mse'][-1]:.4f}"
                  f"  val_mse={val_mse:.4f}  val_mae={val_mae:.4f}")

    # ---------------- Final eval in ORIGINAL ADAS13 units ----------------
    model.eval()
    with torch.no_grad():
        pred_all = model(coeffs_all, t_eval).squeeze(-1).cpu().numpy()
    y_all = ys.cpu().numpy()
    m_all = masks.cpu().numpy()

    pred_adas = pred_all * sigma + mu
    true_adas = y_all * sigma + mu

    def metrics_for(idx_set):
        idx_np = idx_set.cpu().numpy()
        p = pred_adas[idx_np][m_all[idx_np]]
        t = true_adas[idx_np][m_all[idx_np]]
        if len(t) == 0:
            return {}
        resid = p - t
        mae = float(np.mean(np.abs(resid)))
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-8
        r2 = 1.0 - ss_res / ss_tot
        # Baseline: predict train mean
        baseline_rmse = float(np.sqrt(np.mean((t - mu) ** 2)))
        return {"mae": mae, "rmse": rmse, "r2": r2, "baseline_rmse": baseline_rmse,
                "n": int(len(t))}

    train_m = metrics_for(train_idx)
    val_m = metrics_for(val_idx)
    print("TRAIN:", train_m)
    print(" VAL :", val_m)

    # ---------------- Plots ----------------
    # 1) Training curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_mse"], label="train")
    ax[0].plot(history["val_mse"], label="val")
    ax[0].set_title("Neural CDE MSE (normalised units)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(history["val_mae"], color="C2")
    ax[1].set_title("Validation MAE (normalised units)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("MAE"); ax[1].grid(True)
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/cde_training_curves.png", dpi=150); plt.close()

    # 2) Patient trajectories (val)
    plt.figure(figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    shown = 0
    for k, i in enumerate(val_idx.cpu().numpy().tolist()):
        meta = metas[i]; T = len(meta["t"])
        if T < 3: continue
        t_grid = t_eval.cpu().numpy()[:T]
        plt.plot(meta["t"], true_adas[i][:T], "-o", color=colors[shown % 10],
                 label=f"true RID {meta['rid']}" if shown < 4 else None)
        plt.plot(meta["t"], pred_adas[i][:T], "--x", color=colors[shown % 10],
                 label=f"pred RID {meta['rid']}" if shown < 4 else None)
        shown += 1
        if shown >= 6: break
    plt.xlabel("Months since baseline"); plt.ylabel("ADAS13")
    plt.title("Neural CDE longitudinal predictions (validation patients)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cde_trajectories.png", dpi=150); plt.close()

    # 3) Scatter predicted-vs-true (val)
    vp = pred_adas[val_idx.cpu().numpy()][m_all[val_idx.cpu().numpy()]]
    vt = true_adas[val_idx.cpu().numpy()][m_all[val_idx.cpu().numpy()]]
    lo, hi = min(vp.min(), vt.min()) - 1, max(vp.max(), vt.max()) + 1
    plt.figure(figsize=(6, 6))
    plt.scatter(vt, vp, alpha=0.6, s=18)
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("True ADAS13"); plt.ylabel("Predicted ADAS13")
    plt.title(f"CDE predictions (val)  MAE={val_m.get('mae', float('nan')):.2f}"
              f"  R2={val_m.get('r2', float('nan')):.2f}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cde_scatter.png", dpi=150); plt.close()

    # 4) Residual histogram (val)
    plt.figure(figsize=(7, 4))
    plt.hist(vp - vt, bins=40, color="C3", edgecolor="k")
    plt.axvline(0, color="k", lw=1)
    plt.xlabel("Prediction error (pred - true)"); plt.ylabel("count")
    plt.title("Neural CDE validation residuals")
    plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/cde_error_hist.png", dpi=150); plt.close()

    # ---------------- Persist ----------------
    np.savez(
        f"{RESULTS_DIR}/cde_predictions.npz",
        pred=pred_adas, true=true_adas, mask=m_all,
        rid=np.array(rid_order), is_train=is_train.numpy(),
    )
    metrics = {
        "train": train_m, "val": val_m,
        "history": history, "epochs": EPOCHS, "hidden": HIDDEN,
        "input_channels": int(in_ch),
    }
    with open(f"{RESULTS_DIR}/cde_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Neural CDE done. Wrote results/cde_*")


if __name__ == "__main__":
    main()
