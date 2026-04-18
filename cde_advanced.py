import torch
import torchcde
import numpy as np
import matplotlib.pyplot as plt
import json

fm_embs = np.load('adni_data_v2/fm_embeddings.npy')
yolo_embs = np.load('adni_data_v2/yolo_embeddings.npy')
targets = np.load('adni_data_v2/targets.npy')
times = np.load('adni_data_v2/times.npy')
rids = np.load('adni_data_v2/rids.npy')

with open('adni_data_v2/splits.json', 'r') as f:
    splits = json.load(f)
train_rids = splits.get('train_rids', [])

features = np.concatenate([fm_embs, yolo_embs], axis=1) 

unique_rids = np.unique(rids)
patient_seqs = {}
for rid in unique_rids:
    idx = rids == rid
    t = times[idx]
    
    sorted_idx = np.argsort(t)
    t = t[sorted_idx]
    
    t_clean = []
    f_clean = []
    y_clean = []
    last_t = -1
    for i in range(len(t)):
        if t[i] > last_t:
            t_clean.append(t[i])
            f_clean.append(features[idx][sorted_idx][i])
            y_clean.append(targets[idx][sorted_idx][i])
            last_t = t[i]
        else:
            t_clean.append(last_t + 0.5)
            f_clean.append(features[idx][sorted_idx][i])
            y_clean.append(targets[idx][sorted_idx][i])
            last_t += 0.5
            
    patient_seqs[rid] = {
        'times': torch.tensor(t_clean).float(),
        'features': torch.tensor(np.array(f_clean)).float(),
        'targets': torch.tensor(np.array(y_clean)).float()
    }

class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.func = torch.nn.Linear(hidden_channels, hidden_channels * input_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

    def forward(self, x_cde):
        z0 = self.initial(x_cde.evaluate(x_cde.interval[0]))
        class F(torch.nn.Module):
            def __init__(self, func, input_channels, hidden_channels):
                super().__init__()
                self.func = func
                self.input_channels, self.hidden_channels = input_channels, hidden_channels
            def forward(self, t, z):
                return self.func(z).view(z.size(0), self.hidden_channels, self.input_channels)
        z_t = torchcde.cdeint(X=x_cde, z0=z0, func=F(self.func, self.input_channels, self.hidden_channels), t=x_cde.grid_points, method='euler', options={'step_size': 0.5})
        return self.readout(z_t)

model = NeuralCDE(input_channels=97, hidden_channels=32, output_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("Training Batched Longitudinal Neural CDE...")
for epoch in range(50):
    total_loss = 0
    for rid, seq in patient_seqs.items():
        if int(rid) not in train_rids: continue
        if len(seq['times']) < 2: continue
        X = torch.cat([seq['times'].unsqueeze(-1), seq['features']], dim=1).unsqueeze(0)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
        X_cde = torchcde.CubicSpline(coeffs)
        
        pred = model(X_cde)
        y = seq['targets'].unsqueeze(0).unsqueeze(-1)
        loss = torch.nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

plt.figure(figsize=(12, 7))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
c_idx = 0
for rid, seq in patient_seqs.items():
    if int(rid) in train_rids or len(seq['times']) < 2: continue
    X = torch.cat([seq['times'].unsqueeze(-1), seq['features']], dim=1).unsqueeze(0)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
    X_cde = torchcde.CubicSpline(coeffs)
    pred_np = model(X_cde).detach().numpy()[0, :, 0]
    t = seq['times'].numpy()
    y = seq['targets'].numpy()
    
    plt.plot(t, y, label=f'True RID {int(rid)}' if c_idx < 3 else "", color=colors[c_idx%10], marker='o')
    plt.plot(t, pred_np, label=f'Pred RID {int(rid)}' if c_idx < 3 else "", color=colors[c_idx%10], marker='x', linestyle='--')
    c_idx += 1
    if c_idx > 4: break

plt.xlabel('Months since baseline')
plt.ylabel('ADAS13 Cognitive Score')
plt.title('Independent Patient Trajectories: CDE Advanced Validation')
plt.legend()
plt.grid(True)
plt.savefig('adni_data_v2/cde_longitudinal.png')
print("Completed CDE evaluation!")
