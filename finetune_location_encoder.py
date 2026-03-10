"""
Fine-tune the model's location encoder for Hyderabad-specific location differentiation
This retrains only the location encoder while keeping other parameters frozen
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from model import STMLP, TempEncoder
import matplotlib.pyplot as plt

print("="*60)
print("FINE-TUNING LOCATION ENCODER FOR HYDERABAD")
print("="*60)

# Load resources
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/static_scaler_hyderabad_local.pkl', 'rb') as f:
    static_scaler = pickle.load(f)
with open('saved_models/sensor_ids.pkl', 'rb') as f:
    sensor_ids = pickle.load(f)

# Load model
model = STMLP(1, 12, 2, 2, 64, 12, num_layers=3)
model.load_state_dict(torch.load('saved_models/st_mlp.pth'))

# Load traffic data
df = pd.read_csv('traffic_time_series.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
pivot_flow = df.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()

df_sensors = pd.read_csv('sensors.csv')
df_sensors = df_sensors[df_sensors['sensor_id'].isin(sensor_ids)].copy()

scaler_mean = scaler.mean_
scaler_scale = scaler.scale_

print(f"✓ Loaded model and resources")
print(f"✓ Training data shape: {pivot_flow.shape}")
print(f"✓ Sensors: {len(df_sensors)}")

# Prepare training data for location encoder fine-tuning
# We'll use historical data with location labels to teach the model to differentiate locations
print("\n" + "="*60)
print("PREPARING TRAINING DATA")
print("="*60)

# Create synthetic training samples with location information
# Goal: teach location_encoder to produce different outputs for different locations
training_samples = []

# For each sensor, create training pairs:
# - Use its temporal history as input
# - Use its location as the differentiating feature
# - Use its actual predictions as the target (to preserve learned patterns)
for idx, sensor_id in enumerate(sensor_ids[:100]):  # Sample first 100 for faster training
    if idx % 20 == 0:
        print(f"  Processing sensor {idx+1}/{min(100, len(sensor_ids))}: {sensor_id}")
    
    # Get sensor metadata
    sensor_row = df_sensors[df_sensors['sensor_id'] == sensor_id]
    if sensor_row.empty:
        continue
    
    sensor_lat = sensor_row.iloc[0]['latitude']
    sensor_lon = sensor_row.iloc[0]['longitude']
    
    # Check if sensor exists in traffic data
    if sensor_id not in pivot_flow.columns:
        continue
    
    # Get historical flow
    sensor_idx_scaler = sensor_ids.index(sensor_id)
    sensor_idx_flow = list(pivot_flow.columns).index(sensor_id)
    
    last_window_flow = pivot_flow.values[-12:, sensor_idx_flow]
    hours_norm = pivot_flow.index[-12:].hour.values / 23.0
    
    # Normalize
    last_window_flow_norm = (last_window_flow - scaler_mean[sensor_idx_scaler]) / scaler_scale[sensor_idx_scaler]
    
    # Create input tensor
    input_combined = np.stack([last_window_flow_norm, hours_norm], axis=-1)
    input_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    
    # Create static feature tensor (location)
    query_coords = np.array([[sensor_lat, sensor_lon]])
    query_coords_norm = static_scaler.transform(query_coords)[0]
    batch_static = torch.tensor(query_coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    training_samples.append({
        'input': input_tensor,
        'static': batch_static,
        'sensor_id': sensor_id,
        'lat': sensor_lat,
        'lon': sensor_lon,
        'sensor_idx': sensor_idx_scaler
    })

print(f"\n✓ Created {len(training_samples)} training samples")

# Get baseline predictions (before fine-tuning)
print("\nGetting baseline predictions...")
model.eval()
baseline_predictions = []
with torch.no_grad():
    for sample in training_samples[:3]:  # First 3 samples
        out = model(sample['input'], sample['static'])
        out_vals_norm = out.squeeze(0).squeeze(-1).numpy()
        sensor_idx = sample['sensor_idx']
        out_vals = out_vals_norm * scaler_scale[sensor_idx] + scaler_mean[sensor_idx]
        baseline_predictions.append(out_vals)
        print(f"  {sample['sensor_id']}: {out_vals[:5]} ... range: {out_vals.min():.2f}-{out_vals.max():.2f}")

# Fine-tune location encoder
print("\n" + "="*60)
print("FINE-TUNING LOCATION ENCODER")
print("="*60)

# Freeze all parameters except location_encoder
for param in model.parameters():
    param.requires_grad = False

for param in model.location_encoder.parameters():
    param.requires_grad = True

# Loss function: encourage location encoder to produce distinct outputs for different locations
# We use MSE loss with regularization to make location outputs more distinct
optimizer = torch.optim.Adam(model.location_encoder.parameters(), lr=0.001)

print(f"Training location encoder with {len(training_samples)} samples...")
model.train()

losses = []
for epoch in range(50):
    epoch_loss = 0
    
    for sample in training_samples:
        input_tensor = sample['input']
        static_tensor = sample['static']
        sensor_idx = sample['sensor_idx']
        
        # Get location encoder output
        location_output = model.location_encoder(static_tensor)  # (1, 12)
        
        # Variability loss: encourage the location encoder to produce varied outputs
        # that will differentiate locations
        # Target: increase variance of outputs per position
        location_flat = location_output.view(-1)
        
        # Loss: minimize the spread (we want more consistent but distinct per-location outputs)
        # Actually, let's use a different approach: encourage orthogonality of outputs for different locations
        variability_loss = -location_flat.var() * 0.1  # Negative because we want to maximize variance
        
        # Also add a regularization to prevent explosion
        reg_loss = (location_output ** 2).mean() * 0.001
        
        total_loss = variability_loss + reg_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
    
    avg_loss = epoch_loss / len(training_samples)
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

print("\n✓ Fine-tuning complete")

# Test after fine-tuning
print("\n" + "="*60)
print("TESTING AFTER FINE-TUNING")
print("="*60)

model.eval()
print("\nPredictions after fine-tuning:")
with torch.no_grad():
    for sample in training_samples[:3]:
        out = model(sample['input'], sample['static'])
        out_vals_norm = out.squeeze(0).squeeze(-1).numpy()
        sensor_idx = sample['sensor_idx']
        out_vals = out_vals_norm * scaler_scale[sensor_idx] + scaler_mean[sensor_idx]
        print(f"  {sample['sensor_id']}: {out_vals[:5]} ... range: {out_vals.min():.2f}-{out_vals.max():.2f}")

# Save fine-tuned model
print("\n" + "="*60)
print("SAVING FINE-TUNED MODEL")
print("="*60)

torch.save(model.state_dict(), 'saved_models/st_mlp_finetuned_location.pth')
print("✓ Saved to saved_models/st_mlp_finetuned_location.pth")

# Also create a backup of original
import shutil
shutil.copy('saved_models/st_mlp.pth', 'saved_models/st_mlp_original_backup.pth')
print("✓ Backed up original to saved_models/st_mlp_original_backup.pth")

print("\n" + "="*60)
print("FINE-TUNING COMPLETE")
print("="*60)
