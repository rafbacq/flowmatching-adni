import torch
import torchcde
import numpy as np
import matplotlib.pyplot as plt

fm_embs = np.load('adni_data/fm_embeddings.npy')
yolo_embs = np.load('adni_data/yolo_embeddings.npy')
targets = np.load('adni_data/targets.npy')
times = np.arange(len(targets), dtype=np.float32)

N = len(fm_embs)
features = np.concatenate([fm_embs, yolo_embs], axis=1) # (N, 64)

# Format for CDE: (batch, seq_len, channels)
times_tensor = torch.tensor(times).float().unsqueeze(-1)
features_tensor = torch.tensor(features).float()
X = torch.cat([times_tensor, features_tensor], dim=1).unsqueeze(0) # (1, N, 65)

# Neural CDE Requires Cubic Spline representation
coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
X_cde = torchcde.CubicSpline(coeffs)

class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.func = torch.nn.Linear(hidden_channels, hidden_channels * input_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self, x_cde):
        z0 = self.initial(x_cde.evaluate(x_cde.interval[0]))
        class F(torch.nn.Module):
            def __init__(self, func, input_channels, hidden_channels):
                super().__init__()
                self.func = func
                self.input_channels = input_channels
                self.hidden_channels = hidden_channels
            def forward(self, t, z):
                return self.func(z).view(z.size(0), self.hidden_channels, self.input_channels)
        
        z_t = torchcde.cdeint(X=x_cde, z0=z0, func=F(self.func, self.input_channels, self.hidden_channels), t=x_cde.grid_points, method='euler', options={'step_size': 0.5})
        return self.readout(z_t)

model = NeuralCDE(input_channels=65, hidden_channels=32, output_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

targets_tensor = torch.tensor(targets).float().unsqueeze(0).unsqueeze(-1)

print("Training Neural CDE predictor...")
for epoch in range(25):
    pred = model(X_cde)
    loss = torch.nn.functional.mse_loss(pred, targets_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pred_np = pred.detach().numpy()[0, :, 0]
targets_np = targets_tensor.numpy()[0, :, 0]

plt.figure(figsize=(10, 6))
plt.plot(times, targets_np, label='True ADAS13', marker='o', linewidth=2)
plt.plot(times, pred_np, label='Neural CDE Predicted ADAS13', marker='x', linestyle='--', linewidth=2)
plt.xlabel('Months since baseline')
plt.ylabel('ADAS13 Cognitive Score')
plt.title('Patient Trajectory: Neural CDE ADAS13 Predictions from Optical Flow & YOLO')
plt.legend()
plt.grid(True)
plt.savefig('neural_cde_graphs.png')
print("Saved performance charts to neural_cde_graphs.png")
