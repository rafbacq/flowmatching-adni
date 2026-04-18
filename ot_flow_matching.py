import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

class OTUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Linear(1, 16)
        
        self.enc1 = nn.Sequential(nn.Conv2d(2 + 16, 32, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottle = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())
        
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 2, 3, padding=1))
        
    def forward(self, x, t):
        B, _, H, W = x.shape
        t_emb = self.time_embed(t).view(B, 16, 1, 1).expand(B, 16, H, W)
        x_in = torch.cat([x, t_emb], dim=1)
        
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottle(self.pool2(e2))
        
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        out = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return out, b.mean(dim=[2,3])

flows = np.load('adni_data_v2/flows.npy') 
x1 = torch.tensor(flows).permute(0, 3, 1, 2).float() 

model = OTUNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training OT Flow Matching Generator...")
B_full = x1.size(0)
dataset = torch.utils.data.TensorDataset(x1)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(150):
    for (batch_x1,) in loader:
        batch_x1 = batch_x1.cuda()
        optimizer.zero_grad()
        B = batch_x1.size(0)
        x0 = torch.randn_like(batch_x1)
        t = torch.rand(B, 1, device=batch_x1.device)
        t_expand = t.view(B, 1, 1, 1)
        
        x_t = t_expand * batch_x1 + (1 - t_expand) * x0
        target_v = batch_x1 - x0
        
        pred_v, _ = model(x_t, t)
        loss = nn.functional.mse_loss(pred_v, target_v)
        loss.backward()
        optimizer.step()

embeds = []
x1_cuda = x1.cuda()
with torch.no_grad():
    for x in x1_cuda:
        x = x.unsqueeze(0)
        _, e = model(x, torch.ones(1, 1).cuda())
        embeds.append(e.cpu().numpy())

np.save('adni_data_v2/fm_embeddings.npy', np.concatenate(embeds, axis=0))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(np.linalg.norm(x1[0].numpy().transpose(1,2,0), axis=2))
axs[0].set_title("True Optical Flow Mapping")
pred_v_example, _ = model(torch.randn(1, 2, 64, 64).cuda(), torch.ones(1, 1).cuda())
axs[1].imshow(np.linalg.norm(pred_v_example[0].cpu().detach().numpy().transpose(1,2,0), axis=2))
axs[1].set_title("OT Flow Output from Noise")
plt.savefig('adni_data_v2/fm_vector_field.png')
print("OT Flow Matching Complete!")
