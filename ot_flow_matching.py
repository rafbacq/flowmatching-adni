"""
ot_flow_matching.py – Rectified-Flow (Optimal Transport) generative model.
Learns v_θ(x_t, t) to transport N(0,I) → optical-flow distribution.
Outputs:
  • fm_embeddings.npy   – UNet bottleneck features per sample
  • fm_vector_field.png – true-vs-generated comparison grid
  • fm_loss_curve.png   – training loss curve
"""

import torch, torch.nn as nn, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os

DATA = os.environ.get("DATA_DIR", "adni_data")
EPOCHS = int(os.environ.get("FM_EPOCHS", "50"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── architecture ─────────────────────────────────────────────────────────
class OTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 16))
        self.enc1 = nn.Sequential(nn.Conv2d(2 + 16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.pool = nn.MaxPool2d(2)
        self.bottle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.up = nn.Upsample(scale_factor=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.dec2 = nn.Sequential(nn.Conv2d(64, 2, 3, padding=1))

    def forward(self, x, t):
        B, _, H, W = x.shape
        te = self.time_embed(t).view(B, 16, 1, 1).expand(B, 16, H, W)
        x_in = torch.cat([x, te], 1)
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool(e1))
        b = self.bottle(self.pool(e2))
        d1 = self.dec1(torch.cat([self.up(b), e2], 1))
        out = self.dec2(torch.cat([self.up(d1), e1], 1))
        return out, b.mean(dim=[2, 3])  # (pred_v, bottleneck_embed)

# ── data ─────────────────────────────────────────────────────────────────
flows = np.load(f"{DATA}/flows.npy")
x1 = torch.tensor(flows, dtype=torch.float32).permute(0, 3, 1, 2)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x1), batch_size=min(32, len(x1)), shuffle=True
)

model = OTUNet().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

# ── train ────────────────────────────────────────────────────────────────
loss_hist = []
print(f"[FM] Training OT flow matching for {EPOCHS} epochs on {len(x1)} samples …")
for ep in range(EPOCHS):
    ep_loss = 0.0
    for (bx1,) in loader:
        bx1 = bx1.to(DEVICE)
        B = bx1.size(0)
        x0 = torch.randn_like(bx1)
        t = torch.rand(B, 1, device=DEVICE)
        xt = t.view(B, 1, 1, 1) * bx1 + (1 - t.view(B, 1, 1, 1)) * x0
        target_v = bx1 - x0
        pred_v, _ = model(xt, t)
        loss = nn.functional.mse_loss(pred_v, target_v)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item()
    sched.step()
    avg = ep_loss / len(loader)
    loss_hist.append(avg)
    if (ep + 1) % max(1, EPOCHS // 10) == 0:
        print(f"  epoch {ep+1:4d}/{EPOCHS}  loss={avg:.6f}")

# ── embeddings ───────────────────────────────────────────────────────────
embeds = []
model.eval()
with torch.no_grad():
    for i in range(0, len(x1), 64):
        batch = x1[i : i + 64].to(DEVICE)
        _, e = model(batch, torch.ones(batch.size(0), 1, device=DEVICE))
        embeds.append(e.cpu().numpy())
np.save(f"{DATA}/fm_embeddings.npy", np.concatenate(embeds))
print(f"[FM] Saved embeddings → {DATA}/fm_embeddings.npy  shape={np.concatenate(embeds).shape}")

# ── graphs ───────────────────────────────────────────────────────────────
# 1) loss curve
plt.figure(figsize=(8, 4))
plt.plot(loss_hist, linewidth=1.5)
plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("OT Flow Matching – Training Loss")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{DATA}/fm_loss_curve.png", dpi=150)
plt.close()

# 2) true vs generated comparison (2×4 grid)
N_SHOW = min(4, len(x1))
fig, axes = plt.subplots(2, N_SHOW, figsize=(4 * N_SHOW, 8))
if N_SHOW == 1:
    axes = axes.reshape(2, 1)
model.eval()
with torch.no_grad():
    noise = torch.randn(N_SHOW, 2, 64, 64, device=DEVICE)
    gen, _ = model(noise, torch.ones(N_SHOW, 1, device=DEVICE))
for j in range(N_SHOW):
    true_mag = np.linalg.norm(x1[j].numpy().transpose(1, 2, 0), axis=2)
    gen_mag = np.linalg.norm(gen[j].cpu().numpy().transpose(1, 2, 0), axis=2)
    axes[0, j].imshow(true_mag, cmap="inferno"); axes[0, j].set_title(f"True #{j}")
    axes[0, j].axis("off")
    axes[1, j].imshow(gen_mag, cmap="inferno"); axes[1, j].set_title(f"Generated #{j}")
    axes[1, j].axis("off")
fig.suptitle("OT Flow Matching – True vs Generated Flow Fields", fontsize=14)
plt.tight_layout(); plt.savefig(f"{DATA}/fm_vector_field.png", dpi=150); plt.close()
print("[FM] Done.")
