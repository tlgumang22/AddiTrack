import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

# ----------------------------
# 1. Load and prepare data
# ----------------------------
EXCEL_PATH = r"data\all_folders_results.xlsx"
MODEL_PATH = r"models\ivf_inverse_model.pth"
OUTPUT_PATH = r"data\predicted_iv.xlsx"

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
# Outputs = Voltage + Current (removed FeedRate)
y = df[["Voltage", "Current"]].values.astype(np.float32)

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
            nn.Linear(64, 2)   # output: [V, I]
        )
    def forward(self, x):
        return self.net(x)

model = IVFInverseModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 3. Train model (only once)
# ----------------------------
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    print("Training new model...")
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")

# ----------------------------
# 4. Predict IVF for constant Height every 10s
# ----------------------------
target_height = 5.0  # example: keep height = 5mm
times = np.arange(0, 101, 10, dtype=np.float32)  # every 10s till 100s
X_test = np.column_stack([np.full_like(times, target_height), times])

X_test_tensor = torch.tensor(X_test)
predicted_iv = model(X_test_tensor).detach().numpy()

# Save predictions into Excel
out_df = pd.DataFrame(predicted_iv, columns=["Voltage (V)", "Current (A)"])
out_df.insert(0, "Time (s)", times)

out_df.to_excel(OUTPUT_PATH, index=False)

print("\nPredicted IV parameters saved to:", OUTPUT_PATH)
print(out_df.to_string(index=False))
