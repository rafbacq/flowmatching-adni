"""
Optimal-Transport Flow Matching on MRI optical-flow images.

We learn a velocity field v_theta(x_t, t) such that the straight-line path
x_t = (1-t) * x0 + t * x1 has velocity x1 - x0 (rectified / OT flow matching).
At t=1 the bottleneck of the UNet gives us a per-sample embedding that the
downstream Neural CDE consumes.

This script replaces the old `flow_matching_model.py` (which trained with
`.norm()` as the loss and collapsed embeddings to zero) and keeps the v3
data directory.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = "adni_data_v3"
RESULTS_DIR = "results"
EPOCHS = int(os.environ.get("FM_EPOCHS", 80))
BATCH_SIZE = int(os.environ.get("FM_BATCH", 32))
LR = float(os.environ.get("FM_LR", 2e-3))
EMBED_DIM = 64  # size of the bottleneck feature vector


class OTUNet(nn.Module):
    """Small UNet with time conditioning. Outputs (velocity, embedding)."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32)
        )
        self.enc1 = nn.Sequential(nn.Conv2d(2 + 32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.bottle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
        )
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
        self.out = nn.Conv2d(32, 2, 3, padding=1)
        self.embed_head = nn.Linear(128, embed_dim)

    def forward(self, x, t):
        B, _, H, W = x.shape
        t_emb = self.time_embed(t).view(B, -1, 1, 1).expand(B, -1, H, W)
        e1 = self.enc1(torch.cat([x, t_emb], dim=1))
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottle(F.max_pool2d(e2, 2))
        d1 = self.dec1(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d1, scale_factor=2, mode="nearest"), e1], dim=1))
        v = self.out(d2)
        z = self.embed_head(b.mean(dim=[2, 3]))
        return v, z


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {device}")

    flows = np.load(f"{DATA_DIR}/flows.npy")  # (N, H, W, 2)
    x = torch.tensor(flows).permute(0, 3, 1, 2).float()
    # Normalise so the velocity target has unit scale-ish.
    std = x.std().clamp(min=1e-6)
    x = x / std
    print(f"Flow tensor: {tuple(x.shape)}  std(before norm)={std.item():.3f}")

    model = OTUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x), batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    history = []
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for (x1,) in loader:
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=device)
            t_ = t.view(-1, 1, 1, 1)
            x_t = (1 - t_) * x0 + t_ * x1
            target = x1 - x0
            v_pred, _ = model(x_t, t)
            loss = F.mse_loss(v_pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{EPOCHS}  OT-FM loss={mean_loss:.4f}")

    # --------------------------- Extract embeddings ---------------------------
    model.eval()
    embeds = []
    with torch.no_grad():
        for (x1,) in torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x), batch_size=BATCH_SIZE, shuffle=False
        ):
            x1 = x1.to(device)
            _, z = model(x1, torch.ones(x1.size(0), 1, device=device))
            embeds.append(z.cpu().numpy())
    embeds = np.concatenate(embeds, axis=0).astype(np.float32)
    np.save(f"{DATA_DIR}/fm_embeddings.npy", embeds)
    print(f"Saved FM embeddings: {embeds.shape}")

    # --------------------------- Diagnostics / plots --------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("OT-FM velocity MSE")
    plt.title("OT Flow Matching training loss")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fm_training_loss.png", dpi=150)
    plt.close()

    # Visualise a few integrated samples: start from noise, do Euler integration.
    model.eval()
    steps = 20
    with torch.no_grad():
        z = torch.randn(4, 2, x.shape[2], x.shape[3], device=device)
        for k in range(steps):
            t = torch.full((z.size(0), 1), k / steps, device=device)
            v, _ = model(z, t)
            z = z + v / steps
        samples = z.cpu().numpy()
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    for i in range(4):
        real = np.linalg.norm(x[i].numpy().transpose(1, 2, 0), axis=2)
        gen = np.linalg.norm(samples[i].transpose(1, 2, 0), axis=2)
        axs[0, i].imshow(real, cmap="magma"); axs[0, i].set_title(f"real flow |v| #{i}")
        axs[0, i].axis("off")
        axs[1, i].imshow(gen, cmap="magma"); axs[1, i].set_title(f"OT-FM sample |v| #{i}")
        axs[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/fm_samples.png", dpi=150)
    plt.close()

    with open(f"{RESULTS_DIR}/fm_metrics.json", "w") as f:
        json.dump({
            "final_loss": history[-1],
            "min_loss": float(min(history)),
            "epochs": EPOCHS,
            "embed_dim": EMBED_DIM,
            "n_samples": int(x.size(0)),
        }, f, indent=2)
    print("OT Flow Matching done.")


if __name__ == "__main__":
    main()
