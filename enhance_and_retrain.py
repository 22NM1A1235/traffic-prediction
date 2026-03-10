#!/usr/bin/env python3
"""
Retrain the model with enhanced location encoding.
This version uses:
- Sinusoidal positional encoding for coordinates
- Deeper location encoder (8 layers)
- Increased coordinate augmentation (std=0.35)
- Location-weighted fusion (biased toward location branch)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Import the NEW enhanced model
from model import STMLP

# Configuration
EPOCHS = 5  # Reduced to 5 for faster training
BATCH_SIZE = 256
INPUT_LEN = 12
OUTPUT_LEN = 12
INPUT_DIM = 2
STATIC_DIM = 2
EMBED_DIM = 64
NUM_LAYERS = 3

print("="*70)
print("ENHANCED MODEL TRAINING FOR FINE-GRAINED LOCATION DIFFERENTIATION")
print("="*70)

# Load Training Data from CSVs (same as training.py)
print("\nLoading training data from CSVs...")
df_ts = pd.read_csv('traffic_time_series.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0

pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()

sensor_ids = sorted(pivot_flow.columns)
pivot_flow = pivot_flow[sensor_ids]
pivot_hour = pivot_hour[sensor_ids]

print(f"Loaded {len(pivot_flow)} timesteps")
print(f"Training with {len(sensor_ids)} sensors (Bengaluru + Andhra Pradesh)")

# Normalize Flow
scaler = StandardScaler()
flow_values = scaler.fit_transform(pivot_flow.values)
hour_values = pivot_hour.values

data_combined = np.stack([flow_values, hour_values], axis=-1)

# Load Sensors Metadata
df_sensors = pd.read_csv('sensors.csv')
df_sensors = df_sensors.set_index('sensor_id').reindex(sensor_ids).reset_index()
static_feats = df_sensors[['latitude', 'longitude']].values
static_scaler = StandardScaler()
static_feats_norm = static_scaler.fit_transform(static_feats)

print(f"Normalized {len(sensor_ids)} sensor coordinates")

# Load Sequences and create training dataset (single-node mode)
df_seq = pd.read_csv('traffic_sequences.csv')
train_indices = df_seq[df_seq['dataset_split'] == 'train']['history_start_step'].unique()

# Create dataset (same logic as training.py single-node mode)
train_x_list, train_y_list, static_feat_list = [], [], []

for sensor_idx in range(len(sensor_ids)):
    for start_step in train_indices:
        if start_step + INPUT_LEN + OUTPUT_LEN <= len(data_combined):
            x_window = data_combined[start_step:start_step + INPUT_LEN, sensor_idx, :]  # (INPUT_LEN, 2)
            y_window = flow_values[start_step + INPUT_LEN:start_step + INPUT_LEN + OUTPUT_LEN, sensor_idx]  # (OUTPUT_LEN,)
            
            train_x_list.append(x_window)
            train_y_list.append(y_window)
            static_feat_list.append(static_feats_norm[sensor_idx])  # (2,)

train_x = torch.tensor(np.array(train_x_list), dtype=torch.float32).unsqueeze(2)  # (N, INPUT_LEN, 1, 2)
train_y = torch.tensor(np.array(train_y_list), dtype=torch.float32).unsqueeze(-1)  # (N, OUTPUT_LEN, 1)
static_feat = torch.tensor(np.array(static_feat_list), dtype=torch.float32).unsqueeze(1)  # (N, 1, 2)

print(f"Created {len(train_x)} training samples")
print(f"Input shape: {train_x.shape}")
print(f"Output shape: {train_y.shape}")
print(f"Static feature shape: {static_feat.shape}")

# Determine mode (always single-node for per-query predictions)
single_node_mode = True
print(f"✓ Mode: Single-node (per-query predictions)")

# Initialize NEW Enhanced Model
print("\nInitializing ENHANCED model with:")
print("  - Sinusoidal positional encoding for coordinates")
print("  - Deeper location encoder (8 layers with dropout)")
print("  - Location-weighted fusion mechanism")
print("  - Higher coordinate augmentation (std=0.35)")

num_nodes = 1
model = STMLP(
    num_nodes=num_nodes,
    input_len=INPUT_LEN,
    input_dim=INPUT_DIM,
    static_dim=STATIC_DIM,
    embed_dim=EMBED_DIM,
    output_len=OUTPUT_LEN,
    num_layers=NUM_LAYERS
)

# Count parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {num_params:,} trainable parameters")

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training Loop
print(f"\n{'='*70}")
print(f"Training for {EPOCHS} epochs (Batch size: {BATCH_SIZE})")
print(f"{'='*70}\n")

coord_aug_std = 0.35  # Increased for better within-state differentiation
num_samples = len(train_x)

losses_per_epoch = []

model.train()
for epoch in range(EPOCHS):
    permutation = torch.randperm(num_samples)
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, num_samples, BATCH_SIZE):
        indices = permutation[i : i + BATCH_SIZE]
        batch_x = train_x[indices].to(device)
        batch_y = train_y[indices].to(device)
        
        # Handle static features with coordinate augmentation
        curr_batch = batch_x.size(0)
        batch_static = static_feat[indices].clone().to(device)
        
        # Apply coordinate augmentation
        if epoch < EPOCHS * 0.8:
            coord_noise = torch.randn_like(batch_static) * coord_aug_std
            batch_static = batch_static + coord_noise
            batch_static = torch.clamp(batch_static, -3.0, 3.0)
        
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch_x, batch_static)
        loss = criterion(out, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # Print epoch results
    avg_loss = epoch_loss / num_batches
    losses_per_epoch.append(avg_loss)
    
    # Progress indicator
    progress_bar = '█' * int(epoch / EPOCHS * 40) + '░' * (40 - int(epoch / EPOCHS * 40))
    print(f"Epoch {epoch+1:2d}/{EPOCHS} [{progress_bar}] Loss: {avg_loss:.5f}")

print(f"\n{'='*70}")
print("TRAINING COMPLETED")
print(f"{'='*70}")
print(f"\nFinal Loss: {losses_per_epoch[-1]:.5f}")
print(f"Initial Loss: {losses_per_epoch[0]:.5f}")
print(f"Improvement: {(losses_per_epoch[0] - losses_per_epoch[-1]) / losses_per_epoch[0] * 100:.1f}%")

# Save the enhanced model
print("\nSaving enhanced model...")
torch.save(model.state_dict(), 'saved_models/st_mlp.pth')
print(f"✓ Model saved to saved_models/st_mlp.pth")

# Also save updated scalers
print("\nSaving updated scalers...")
with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('saved_models/static_scaler.pkl', 'wb') as f:
    pickle.dump(static_scaler, f)
with open('saved_models/sensor_ids.pkl', 'wb') as f:
    pickle.dump(sensor_ids, f)
print(f"✓ Scalers saved")

print("\n" + "="*70)
print("MODEL READY FOR LOCATION-SPECIFIC PREDICTIONS")
print("="*70)
print("\nKey improvements in this model:")
print("  ✓ Sinusoidal positional encoding for fine-grained location encoding")
print("  ✓ Deeper location encoder (8 layers) for spatial pattern learning")
print("  ✓ Location-weighted fusion (biased toward location branch)")
print("  ✓ Higher coordinate augmentation (std=0.35) for within-state differentiation")
print("  ✓ Dropout regularization (0.1) for better generalization")
print("\nExpected: Different locations in same state → Different predictions")
print("="*70)
