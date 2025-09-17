import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# ----------------------------
# 1. Load and prepare data
# ----------------------------
EXCEL_PATH = "data/all_folders_results.xlsx"
import os
print(os.path.exists(EXCEL_PATH))

df = pd.read_excel(EXCEL_PATH)

df = df.rename(columns={
    "Voltage (V)": "Voltage",
    "Current (A)": "Current",
    "FeedRate (mm/min)": "FeedRate",
    "Time (s)": "Time",
    "Height (mm)": "Height"
})

# Inputs = Height + Time
X = df[["Height", "Time"]].values.astype(np.float32)
# Outputs = IVF
y = df[["Voltage", "Current", "FeedRate"]].values.astype(np.float32)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# ----------------------------
# 2. Define Neural Network
# ----------------------------
class IVFInverseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  # input: [Height, Time]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # output: [V, I, F]
        )
    def forward(self, x):
        return self.net(x)

model = IVFInverseModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 3. Train the model
# ----------------------------
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----------------------------
# 4. Predict IVF for constant Height every 10s
# ----------------------------
target_height = 5.0  # example: keep height = 5mm
times = np.arange(0, 101, 10, dtype=np.float32)  # every 10s till 100s
X_test = np.column_stack([np.full_like(times, target_height), times])

X_test_tensor = torch.tensor(X_test)
predicted_ivf = model(X_test_tensor).detach().numpy()

# Save as table
out_df = pd.DataFrame(predicted_ivf, columns=["Voltage (V)", "Current (A)", "FeedRate (mm/min)"])
out_df.insert(0, "Time (s)", times)

print("\nPredicted IVF parameters to maintain constant height:")
print(out_df.to_string(index=False))