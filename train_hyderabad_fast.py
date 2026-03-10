"""
Fast Hyderabad-only model training. Simplified for speed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from model import STMLP

print("=" * 80)
print("FAST HYDERABAD MODEL TRAINING")
print("=" * 80)

# Load Hyderabad sensors
hyderabad_sensor_ids = []
with open('hyderabad_sensors.txt', 'r') as f:
    hyderabad_sensor_ids = [line.strip() for line in f]

print(f"\n[1/4] Hyderabad sensors: {len(hyderabad_sensor_ids)}")

# Load data - Hyderabad only
print("[2/4] Loading Hyderabad traffic data...")
df_ts = pd.read_csv('traffic_time_series.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0
df_ts = df_ts[df_ts['sensor_id'].isin(hyderabad_sensor_ids)]

pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()

hyderabad_with_data = [s for s in hyderabad_sensor_ids if s in pivot_flow.columns]
print(f"   Sensors with data: {len(hyderabad_with_data)}")

pivot_flow = pivot_flow[hyderabad_with_data]
pivot_hour = pivot_hour[hyderabad_with_data]

# Normalize
scaler = StandardScaler()
flow_norm = scaler.fit_transform(pivot_flow.values)
hour_norm = pivot_hour.values

# Coordinates
df_sensors = pd.read_csv('sensors.csv')
df_sensors = df_sensors[df_sensors['sensor_id'].isin(hyderabad_with_data)].copy()
df_sensors = df_sensors.set_index('sensor_id').reindex(hyderabad_with_data).reset_index()

static_feats = df_sensors[['latitude', 'longitude']].values
static_scaler = StandardScaler()
static_feats_norm = static_scaler.fit_transform(static_feats)

# Training data
print("[3/4] Creating training data...")
INPUT_LEN = 12
OUTPUT_LEN = 12

X_train, Y_train, S_train = [], [], []

df_seq = pd.read_csv('traffic_sequences.csv')
train_indices = df_seq[df_seq['dataset_split'] == 'train']['history_start_step'].unique()

for start_idx in train_indices:
    if start_idx + INPUT_LEN + OUTPUT_LEN <= len(flow_norm):
        for sensor_idx in range(len(hyderabad_with_data)):
            X = np.stack([flow_norm[start_idx:start_idx + INPUT_LEN, sensor_idx],
                         hour_norm[start_idx:start_idx + INPUT_LEN, sensor_idx]], axis=-1)
            Y = flow_norm[start_idx + INPUT_LEN:start_idx + INPUT_LEN + OUTPUT_LEN, sensor_idx]
            S = static_feats_norm[sensor_idx]
            
            X_train.append(X)
            Y_train.append(Y)
            S_train.append(S)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
S_train = np.array(S_train)

print(f"   Training samples: {len(X_train)}")

# Train
print("[4/4] Training model...")
device = torch.device('cpu')

model = STMLP(num_nodes=1, input_len=INPUT_LEN, input_dim=2,
              static_dim=2, embed_dim=64, output_len=OUTPUT_LEN)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Use SGD for speed
criterion = nn.MSELoss()

for epoch in range(2):  # Just 2 epochs
    epoch_loss = 0
    count = 0
    
    # Use subset for speed
    subset_size = min(2000, len(X_train))
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    
    for idx in indices:
        X = torch.tensor(X_train[idx:idx+1], dtype=torch.float32, device=device).unsqueeze(2)
        Y = torch.tensor(Y_train[idx:idx+1], dtype=torch.float32, device=device)
        S = torch.tensor(S_train[idx:idx+1], dtype=torch.float32, device=device).unsqueeze(1)
        
        out = model(X, S).squeeze(-1)
        loss = criterion(out, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        count += 1
        
        if count % 200 == 0:
            print(f"  Epoch {epoch+1} - Batch {count}: Loss {epoch_loss/count:.6f}")

print("\n[SUCCESS] Saving model...")
torch.save(model.state_dict(), 'saved_models/st_mlp.pth')

with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('saved_models/static_scaler.pkl', 'wb') as f:
    pickle.dump(static_scaler, f)
with open('saved_models/sensor_ids.pkl', 'wb') as f:
    pickle.dump(hyderabad_with_data, f)

print("=" * 80)
print("MODEL TRAINED ON HYDERABAD DATA ONLY")
print("=" * 80)
print(f"Sensors: {len(hyderabad_with_data)}")
print(f"Training samples: {len(X_train)}")
print("Model should now be location-sensitive within Hyderabad!")
print("=" * 80 + "\n")
