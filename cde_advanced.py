"""
cde_advanced.py – Batched Neural CDE for longitudinal ADAS13 prediction.
Reads fm_embeddings + yolo_embeddings + targets/times/rids from DATA dir.
Outputs:
  • cde_longitudinal.png  – predicted vs true trajectories (val patients)
  • cde_loss_curve.png    – training loss over epochs
"""

import torch, torchcde, numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.environ.get("DATA_DIR", "adni_data")
EPOCHS = int(os.environ.get("CDE_EPOCHS", "80"))

# ── load data ────────────────────────────────────────────────────────────
fm = np.load(f"{DATA}/fm_embeddings.npy")
yolo = np.load(f"{DATA}/yolo_embeddings.npy")
targets = np.load(f"{DATA}/targets.npy")
times = np.load(f"{DATA}/times.npy")
rids = np.load(f"{DATA}/rids.npy")
with open(f"{DATA}/splits.json") as f:
    splits = json.load(f)
train_rids = set(splits.get("train_rids", []))

features = np.concatenate([fm, yolo], axis=1)
INPUT_CH = features.shape[1] + 1  # +1 for time channel

# ── build per-patient sequences ──────────────────────────────────────────
patient_seqs: dict = {}
for rid in np.unique(rids):
    mask = rids == rid
    t = times[mask]; order = np.argsort(t)
    t = t[order]; feat = features[mask][order]; tgt = targets[mask][order]
    # enforce strictly increasing times
    t_clean, f_clean, y_clean = [], [], []
    last = -1.0
    for i in range(len(t)):
        ti = t[i] if t[i] > last else last + 0.5
        t_clean.append(ti); f_clean.append(feat[i]); y_clean.append(tgt[i])
        last = ti
    patient_seqs[int(rid)] = {
        "t": torch.tensor(t_clean, dtype=torch.float32),
        "f": torch.tensor(np.array(f_clean), dtype=torch.float32),
        "y": torch.tensor(np.array(y_clean), dtype=torch.float32),
    }

# ── model ────────────────────────────────────────────────────────────────
class CDEFunc(torch.nn.Module):
    def __init__(self, h, ic):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(h, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, h * ic),
        )
        self.h, self.ic = h, ic
    def forward(self, t, z):
        return self.net(z).view(z.size(0), self.h, self.ic)

class NeuralCDE(torch.nn.Module):
    def __init__(self, ic, h, oc):
        super().__init__()
        self.initial = torch.nn.Linear(ic, h)
        self.func = CDEFunc(h, ic)
        self.readout = torch.nn.Linear(h, oc)
    def forward(self, X_cde):
        z0 = self.initial(X_cde.evaluate(X_cde.interval[0]))
        z = torchcde.cdeint(X=X_cde, z0=z0, func=self.func,
                            t=X_cde.grid_points, method="euler",
                            options={"step_size": 0.5})
        return self.readout(z)

model = NeuralCDE(ic=INPUT_CH, h=64, oc=1)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, EPOCHS // 3), gamma=0.5)

# ── train ────────────────────────────────────────────────────────────────
loss_hist = []
print(f"[CDE] Training for {EPOCHS} epochs on {sum(1 for r in patient_seqs if r in train_rids)} train patients …")
for ep in range(EPOCHS):
    model.train(); ep_loss = 0.0; n = 0
    for rid, seq in patient_seqs.items():
        if rid not in train_rids or len(seq["t"]) < 2:
            continue
        X = torch.cat([seq["t"].unsqueeze(-1), seq["f"]], dim=1).unsqueeze(0)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
        X_cde = torchcde.CubicSpline(coeffs)
        pred = model(X_cde)
        y = seq["y"].unsqueeze(0).unsqueeze(-1)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item(); n += 1
    sched.step()
    avg = ep_loss / max(n, 1)
    loss_hist.append(avg)
    if (ep + 1) % max(1, EPOCHS // 10) == 0:
        print(f"  epoch {ep+1:4d}/{EPOCHS}  loss={avg:.4f}")

# ── graphs ───────────────────────────────────────────────────────────────
# 1) loss curve
plt.figure(figsize=(8, 4))
plt.plot(loss_hist, linewidth=1.5, color="teal")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Neural CDE – Training Loss")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{DATA}/cde_loss_curve.png", dpi=150); plt.close()

# 2) trajectory predictions (val patients)
model.eval()
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.Set1(np.linspace(0, 1, 10))
ci = 0
for rid, seq in patient_seqs.items():
    if rid in train_rids or len(seq["t"]) < 2:
        continue
    X = torch.cat([seq["t"].unsqueeze(-1), seq["f"]], dim=1).unsqueeze(0)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
    X_cde = torchcde.CubicSpline(coeffs)
    with torch.no_grad():
        pred_np = model(X_cde).numpy()[0, :, 0]
    t_np = seq["t"].numpy(); y_np = seq["y"].numpy()
    ax.plot(t_np, y_np, "o-", color=colors[ci % 10], label=f"True RID {rid}", linewidth=2)
    ax.plot(t_np, pred_np, "x--", color=colors[ci % 10], label=f"Pred RID {rid}", linewidth=1.5)
    ci += 1
    if ci >= 5:
        break
ax.set_xlabel("Months since baseline"); ax.set_ylabel("ADAS13")
ax.set_title("Neural CDE – Predicted vs True Patient Trajectories (Validation)")
if ci > 0:
    ax.legend(fontsize=8)
ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{DATA}/cde_longitudinal.png", dpi=150); plt.close()
print("[CDE] Done.")
