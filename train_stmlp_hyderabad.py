"""
Train STMLP model on Hyderabad-only data for location-sensitive predictions
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
print("TRAINING STMLP ON HYDERABAD DATA")
print("=" * 80)

# Configuration
BATCH_SIZE = 32
EPOCHS = 1  # Just 1 epoch for speed
LEARNING_RATE = 0.001
DEVICE = torch.device('cpu')
INPUT_LEN = 12
OUTPUT_LEN = 12
EMBED_DIM = 64

print(f"\n[1/5] Loading Hyderabad sensors...")

# Load Hyderabad sensors
with open('hyderabad_sensors.txt', 'r') as f:
    hyderabad_ids = [line.strip() for line in f]
print(f"      Found {len(hyderabad_ids)} sensors")

# Load and filter traffic data
print(f"[2/5] Loading Hyderabad traffic data...")
df_ts = pd.read_csv('traffic_time_series.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0
df_ts = df_ts[df_ts['sensor_id'].isin(hyderabad_ids)]

# Pivot data
pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()

# Filter to sensors with data
ids_with_data = [s for s in hyderabad_ids if s in pivot_flow.columns]
print(f"      {len(ids_with_data)} sensors with traffic data")
print(f"      {len(pivot_flow)} timesteps")

pivot_flow = pivot_flow[ids_with_data]
pivot_hour = pivot_hour[ids_with_data]

# Normalize flow data
scaler = StandardScaler()
flow_norm = scaler.fit_transform(pivot_flow.values)
hour_norm = pivot_hour.values

# Load and normalize coordinates
print(f"[3/5] Normalizing coordinates...")
df_sensors = pd.read_csv('sensors.csv')
sensor_coords = []
for s in ids_with_data:
    row = df_sensors[df_sensors['sensor_id'] == s]
    if len(row) > 0:
        sensor_coords.append([row.iloc[0]['latitude'], row.iloc[0]['longitude']])

static_scaler = StandardScaler()
coords_norm = static_scaler.fit_transform(np.array(sensor_coords))

# Create training data
print(f"[4/5] Creating training data...")
X_data, Y_data, S_data = [], [], []

# Use all available data for training
np.random.seed(42)
for start_idx in range(0, len(flow_norm) - INPUT_LEN - OUTPUT_LEN, 5):  # Step by 5 for speed
    for sensor_idx in range(len(ids_with_data)):
        x = np.stack([
            flow_norm[start_idx:start_idx + INPUT_LEN, sensor_idx],
            hour_norm[start_idx:start_idx + INPUT_LEN, sensor_idx]
        ], axis=-1)  # (INPUT_LEN, 2)
        
        y = flow_norm[start_idx + INPUT_LEN:start_idx + INPUT_LEN + OUTPUT_LEN, sensor_idx]
        s = coords_norm[sensor_idx]
        
        X_data.append(x)
        Y_data.append(y)
        S_data.append(s)

X_data = np.array(X_data)
Y_data = np.array(Y_data) 
S_data = np.array(S_data)

print(f"      Created {len(X_data)} training samples")
print(f"      Sensors: {len(ids_with_data)}, Timesteps: {len(flow_norm)}")

# Create model (SINGLE NODE for per-sensor prediction)
print(f"[5/5] Training STMLP model...")
model = STMLP(
    num_nodes=1,  # Single-node model as per app.py configuration
    input_len=INPUT_LEN,
    input_dim=2,
    static_dim=2,
    embed_dim=EMBED_DIM,
    output_len=OUTPUT_LEN,
    num_layers=2  # Reduced for speed
)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0
    batch_count = 0
    
    # Shuffle data
    indices = np.random.permutation(len(X_data))
    
    # Process in batches
    for batch_start in range(0, len(indices), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(indices))
        batch_indices = indices[batch_start:batch_end]
        
        # Get batch data
        X_batch = torch.tensor(X_data[batch_indices], dtype=torch.float32, device=DEVICE)
        Y_batch = torch.tensor(Y_data[batch_indices], dtype=torch.float32, device=DEVICE)
        S_batch = torch.tensor(S_data[batch_indices], dtype=torch.float32, device=DEVICE)
        
        # Reshape for model with single-node architecture
        # X: (Batch, INPUT_LEN, 2) -> (Batch, INPUT_LEN, Nodes=1, 2)
        X_batch = X_batch.unsqueeze(2)
        # S: (Batch, 2) -> (Batch, Nodes=1, 2)
        S_batch = S_batch.unsqueeze(1)
        # Y: (Batch, OUTPUT_LEN) -> (Batch, OUTPUT_LEN, Nodes=1)
        Y_batch = Y_batch.unsqueeze(2)
        
        # Forward
        out = model(X_batch, S_batch)
        loss = criterion(out, Y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        if batch_count % 20 == 0:
            avg_loss = epoch_loss / batch_count
            print(f"  Epoch {epoch+1} - Batch {batch_count}: Loss {avg_loss:.6f}")

# Save model and resources
print(f"\n[SUCCESS] Saving model...")
torch.save(model.state_dict(), 'saved_models/st_mlp.pth')

with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('saved_models/static_scaler.pkl', 'wb') as f:
    pickle.dump(static_scaler, f)
with open('saved_models/sensor_ids.pkl', 'wb') as f:
    pickle.dump(ids_with_data, f)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print(f"Model saved to saved_models/st_mlp.pth")
print(f"Trained on {len(ids_with_data)} Hyderabad sensors")
print(f"Training samples: {len(X_data)}")
print("=" * 80)
