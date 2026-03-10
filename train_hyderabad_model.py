"""
Train model using ONLY Hyderabad sensors.
This specialized training makes the model highly sensitive to location variations within Hyderabad.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from model import STMLP

print("=" * 80)
print("HYDERABAD-FOCUSED MODEL TRAINING")
print("=" * 80)

# Configuration
INPUT_LEN = 12
OUTPUT_LEN = 12
INPUT_DIM = 2 
STATIC_DIM = 2
EMBED_DIM = 64
EPOCHS = 5  # Reduced from 10
BATCH_SIZE = 64  # Increased batch size for faster training
REG_LAMBDA = 0.0  # No spatial regularization needed

print("\n[1/5] Loading Hyderabad sensors list...")
hyderabad_sensor_ids = []
with open('hyderabad_sensors.txt', 'r') as f:
    for line in f:
        hyderabad_sensor_ids.append(line.strip())

print(f"   Found {len(hyderabad_sensor_ids)} Hyderabad sensors")

print("\n[2/5] Loading and filtering traffic data...")
# Load traffic data
df_ts = pd.read_csv('traffic_time_series.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0

# Filter to only Hyderabad sensors
print(f"   Original traffic records: {len(df_ts)}")
df_ts_hyderbad = df_ts[df_ts['sensor_id'].isin(hyderabad_sensor_ids)]
print(f"   After filtering to Hyderabad: {len(df_ts_hyderbad)}")

# Pivot
pivot_flow = df_ts_hyderbad.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = df_ts_hyderbad.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()

print(f"   Pivot shape: {pivot_flow.shape}")

# Ensure consistent sensor order - use only Hyderabad sensors
hyderabad_with_data = [s for s in hyderabad_sensor_ids if s in pivot_flow.columns]
print(f"   Hyderabad sensors with complete data: {len(hyderabad_with_data)}")

pivot_flow = pivot_flow[hyderabad_with_data]
pivot_hour = pivot_hour[hyderabad_with_data]

# Normalize flow
scaler = StandardScaler()
flow_values = scaler.fit_transform(pivot_flow.values)
hour_values = pivot_hour.values

# Combine features
data_combined = np.stack([flow_values, hour_values], axis=-1)  # (Time, Nodes, Features=2)

print("\n[3/5] Loading sensor metadata...")
df_sensors = pd.read_csv('sensors.csv')
df_sensors = df_sensors[df_sensors['sensor_id'].isin(hyderabad_with_data)].copy()
df_sensors = df_sensors.set_index('sensor_id').reindex(hyderabad_with_data).reset_index()

static_feats = df_sensors[['latitude', 'longitude']].values
static_scaler = StandardScaler()
static_feats_norm = static_scaler.fit_transform(static_feats)

print(f"   Sensor coordinates normalized")
print(f"   Lat range (normalized): {static_feats_norm[:, 0].min():.4f} to {static_feats_norm[:, 0].max():.4f}")
print(f"   Lon range (normalized): {static_feats_norm[:, 1].min():.4f} to {static_feats_norm[:, 1].max():.4f}")

print("\n[4/5] Creating training batches...")
df_seq = pd.read_csv('traffic_sequences.csv')
train_indices = df_seq[df_seq['dataset_split'] == 'train']['history_start_step'].unique()

X_train, Y_train, S_train = [], [], []

for start_idx in train_indices:
    if start_idx + INPUT_LEN + OUTPUT_LEN <= len(data_combined):
        for sensor_idx in range(len(hyderabad_with_data)):
            X = data_combined[start_idx:start_idx + INPUT_LEN, sensor_idx, :]  # (INPUT_LEN, 2)
            Y = flow_values[start_idx + INPUT_LEN:start_idx + INPUT_LEN + OUTPUT_LEN, sensor_idx]  # (OUTPUT_LEN,)
            S = static_feats_norm[sensor_idx]  # (2,) - location
            
            X_train.append(X)
            Y_train.append(Y)
            S_train.append(S)

X_train = np.array(X_train)  # (N, INPUT_LEN, 2)
Y_train = np.array(Y_train)  # (N, OUTPUT_LEN)
S_train = np.array(S_train)  # (N, 2)

print(f"   Created {len(X_train)} training samples")

print("\n[5/5] Training Hyderabad-focused model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model with single node (per-location predictions)
model = STMLP(num_nodes=1, input_len=INPUT_LEN, input_dim=INPUT_DIM,
              static_dim=STATIC_DIM, embed_dim=EMBED_DIM, output_len=OUTPUT_LEN)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(f"\nTraining on {device}:")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"  Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")

for epoch in range(EPOCHS):
    epoch_loss = 0
    num_batches = 0
    
    # Shuffle
    indices = np.random.permutation(len(X_train))
    
    for batch_idx in range(0, len(X_train), BATCH_SIZE):
        batch_indices = indices[batch_idx:batch_idx + BATCH_SIZE]
        
        X_batch = torch.tensor(X_train[batch_indices], dtype=torch.float32, device=device)  # (B, INPUT_LEN, 2)
        Y_batch = torch.tensor(Y_train[batch_indices], dtype=torch.float32, device=device)  # (B, OUTPUT_LEN)
        S_batch = torch.tensor(S_train[batch_indices], dtype=torch.float32, device=device)  # (B, 2)
        
        # Reshape for model
        X_batch = X_batch.unsqueeze(2)  # (B, INPUT_LEN, 1, 2)
        S_batch = S_batch.unsqueeze(1)  # (B, 1, 2)
        
        # Forward
        output = model(X_batch, S_batch)  # (B, OUTPUT_LEN, 1)
        output = output.squeeze(-1)  # (B, OUTPUT_LEN)
        
        # Loss
        loss = criterion(output, Y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} - Loss: {avg_loss:.6f}")

print("\n[SUCCESS] Model trained on Hyderabad data only!")

# Save model
print("\nSaving trained model and scalers...")
torch.save(model.state_dict(), 'saved_models/st_mlp.pth')

with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('saved_models/static_scaler.pkl', 'wb') as f:
    pickle.dump(static_scaler, f)

with open('saved_models/sensor_ids.pkl', 'wb') as f:
    pickle.dump(hyderabad_with_data, f)

print("   ✓ Model saved to saved_models/st_mlp.pth")
print("   ✓ Scaler saved to saved_models/scaler.pkl")
print("   ✓ Sensor IDs saved to saved_models/sensor_ids.pkl")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nModel trained on {len(hyderabad_with_data)} Hyderabad sensors")
print(f"Total training samples: {len(X_train)}")
print(f"Expected to be highly location-sensitive within Hyderabad region")
print("=" * 80 + "\n")
