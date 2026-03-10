"""
Simplest possible Hyderabad training - just 1 epoch on small subset
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import sys

sys.stdout.flush()

print("=" * 80)
print("ULTRA-FAST HYDERABAD TRAINING (1 EPOCH, SMALL SUBSET)")
print("=" * 80)
sys.stdout.flush()

# Load sensors
with open('hyderabad_sensors.txt', 'r') as f:
    hyderabad_ids = [line.strip() for line in f]

print(f"[1] Sensors: {len(hyderabad_ids)}", flush=True)

# Load data
df_ts = pd.read_csv('traffic_time_series.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0
df_ts = df_ts[df_ts['sensor_id'].isin(hyderabad_ids)]

pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()
ids_with_data = [s for s in hyderabad_ids if s in pivot_flow.columns]

print(f"[2] Traffic data loaded: {len(ids_with_data)} sensors, {len(pivot_flow)} timesteps", flush=True)

# Normalize
scaler = StandardScaler()
flow_norm = scaler.fit_transform(pivot_flow[ids_with_data].values)
hour_norm = pivot_hour[ids_with_data].values

df_sensors = pd.read_csv('sensors.csv')
coord_data = []
for s in ids_with_data:
    row = df_sensors[df_sensors['sensor_id'] == s]
    if len(row) > 0:
        coord_data.append([row.iloc[0]['latitude'], row.iloc[0]['longitude']])

static_scaler = StandardScaler()
coords_norm = static_scaler.fit_transform(np.array(coord_data))

print(f"[3] Data normalized", flush=True)

# Create data - just 1000 samples
X_list, Y_list, S_list = [], [], []
for t in range(100, len(flow_norm) - 24):
    for si in range(min(10, len(ids_with_data))):  # Just 10 sensors
        X = np.stack([flow_norm[t:t+12, si], hour_norm[t:t+12, si]], axis=-1)
        Y = flow_norm[t+12:t+24, si]
        S = coords_norm[si]
        X_list.append(X)
        Y_list.append(Y)
        S_list.append(S)

X_arr = np.array(X_list)
Y_arr = np.array(Y_list)
S_arr = np.array(S_list)

print(f"[4] Training samples: {len(X_arr)}", flush=True)

# Create and train simple linear model (faster for POC)
class FastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12*2 + 2, 128)  # input_len*2 + coords
        self.fc2 = nn.Linear(128, 12)
        
    def forward(self, x, s):
        batch = x.shape[0]
        x = x.reshape(batch, -1)  # (batch, 12*2)
        x = torch.cat([x, s], dim=1)  # (batch, 12*2+2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = FastModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("[5] Training (1 epoch on 100 samples)...", flush=True)

for i in range(min(100, len(X_arr))):
    x = torch.tensor(X_arr[i:i+1], dtype=torch.float32)
    y = torch.tensor(Y_arr[i:i+1], dtype=torch.float32)
    s = torch.tensor(S_arr[i:i+1], dtype=torch.float32)
    
    out = model(x, s)
    loss = criterion(out, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 25 == 0:
        print(f"  Batch {i}: Loss {loss.item():.6f}", flush=True)

print("[6] Saving model...", flush=True)

# Save
state = model.state_dict()
torch.save(state, 'saved_models/st_mlp.pth')

with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('saved_models/static_scaler.pkl', 'wb') as f:
    pickle.dump(static_scaler, f)
with open('saved_models/sensor_ids.pkl', 'wb') as f:
    pickle.dump(ids_with_data, f)

print("\n" + "=" * 80)
print("HYDERABAD MODEL TRAINING COMPLETE!")
print("=" * 80)
print("Training completed successfully.")
print("Model is location-aware for Hyderabad region.")
print("=" * 80 + "\n")
sys.stdout.flush()
