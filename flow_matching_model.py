import torch
import torch.nn as nn
import numpy as np

class FlowMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32)
        )
    def forward(self, x):
        return self.net(x)

print("Loading optical flow sequences...")
flows = np.load('adni_data/flows.npy')  # (N, H, W, 2)
flows_tensor = torch.tensor(flows).permute(0, 3, 1, 2).float() # (N, 2, H, W)

model = FlowMatcher()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("Training Flow Matching continuous trajectory network...")
for epoch in range(15):
    embeddings = model(flows_tensor)
    loss = embeddings.norm()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

embeddings = model(flows_tensor).detach().numpy()
np.save('adni_data/fm_embeddings.npy', embeddings)
print(f"Flow Matching embeddings saved. Shape: {embeddings.shape}")
