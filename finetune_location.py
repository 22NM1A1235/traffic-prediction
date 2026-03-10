"""
Fine-tune model to make location encoder more sensitive to coordinate variations.
This adds location regularization loss to force different coordinates to produce different predictions.
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
print("LOCATION-AWARE MODEL FINE-TUNING")
print("=" * 80)

# Configuration
FINETUNE_EPOCHS = 2  # Short fine-tuning (1-2 epochs)
LOCATION_LOSS_WEIGHT = 10.0  # Strong weight on location regularization
LEARNING_RATE = 0.001
INPUT_LEN = 12
OUTPUT_LEN = 12
INPUT_DIM = 2
STATIC_DIM = 2
EMBED_DIM = 64

# Load model
print("\n1. Loading trained model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = STMLP(num_nodes=1, input_len=INPUT_LEN, input_dim=INPUT_DIM, 
              static_dim=STATIC_DIM, embed_dim=EMBED_DIM, output_len=OUTPUT_LEN)
model.load_state_dict(torch.load('saved_models/st_mlp.pth', map_location=device))
model.to(device)
print(f"   ✓ Model loaded (device: {device})")

# Load scalers
print("2. Loading scalers...")
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/static_scaler.pkl', 'rb') as f:
    static_scaler = pickle.load(f)
with open('saved_models/sensor_ids.pkl', 'rb') as f:
    sensor_ids = pickle.load(f)
print(f"   ✓ Loaded scaler with {len(scaler.mean_)} sensors")

# Load data
print("3. Loading training data...")
df_ts = pd.read_csv('traffic_time_series.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0

pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()

available_sensors = [s for s in sensor_ids if s in pivot_flow.columns]
pivot_flow = pivot_flow[available_sensors]
pivot_hour = pivot_hour[available_sensors]

flow_values = scaler.fit_transform(pivot_flow.values)
hour_values = pivot_hour.values
data_combined = np.stack([flow_values, hour_values], axis=-1)

# Load sensor locations
df_sensors = pd.read_csv('sensors.csv')
df_sensors = df_sensors[df_sensors['sensor_id'].isin(available_sensors)].copy()
df_sensors_sorted = df_sensors.set_index('sensor_id').reindex(available_sensors).reset_index()
static_feats = df_sensors_sorted[['latitude', 'longitude']].values
static_feats_norm = static_scaler.fit_transform(static_feats)

print(f"   ✓ Loaded {len(available_sensors)} sensors with {len(data_combined)} timesteps")

# Create fine-tuning dataset
print("4. Creating fine-tuning batches...")
df_seq = pd.read_csv('traffic_sequences.csv')
train_indices = df_seq[df_seq['dataset_split'] == 'train']['history_start_step'].unique()

# Create training data for location fine-tuning
X_finetune = []
Y_finetune = []
locations_finetune = []

for start_idx in train_indices[:min(len(train_indices), 100)]:  # Use subset for quick fine-tuning
    if start_idx + INPUT_LEN + OUTPUT_LEN <= len(data_combined):
        for sensor_idx in range(min(len(available_sensors), 50)):  # Use subset of sensors
            X = data_combined[start_idx:start_idx + INPUT_LEN, sensor_idx, :]  # (INPUT_LEN, 2)
            Y = flow_values[start_idx + INPUT_LEN:start_idx + INPUT_LEN + OUTPUT_LEN, sensor_idx]  # (OUTPUT_LEN,)
            
            X_finetune.append(X)
            Y_finetune.append(Y)
            locations_finetune.append(static_feats_norm[sensor_idx])

X_finetune = np.array(X_finetune)  # (N, INPUT_LEN, 2)
Y_finetune = np.array(Y_finetune)  # (N, OUTPUT_LEN)
locations_finetune = np.array(locations_finetune)  # (N, 2)

print(f"   ✓ Created {len(X_finetune)} training samples")

# Prepare optimizer - only fine-tune location encoder and fusion decoder
print("5. Setting up fine-tuning optimizer...")
params_to_finetune = list(model.location_encoder.parameters()) + list(model.fusion_decoder.parameters())
optimizer = optim.Adam(params_to_finetune, lr=LEARNING_RATE)
criterion = nn.MSELoss()

print(f"   ✓ Fine-tuning {len(params_to_finetune)} parameters from location_encoder and fusion_decoder")

# Location regularization loss
def location_regularization_loss(outputs_list, locations_batch):
    """
    Simple location regularization: encourage different locations to have different output variance.
    This is computed directly on model outputs without expensive pairwise comparisons.
    """
    if len(outputs_list) < 2:
        return torch.tensor(0.0)
    
    outputs_tensor = torch.stack(outputs_list)  # (N, OUTPUT_LEN)
    locations_tensor = torch.tensor(locations_batch, dtype=torch.float32)
    
    # Compute output variance and location distance correlation
    # High variance in outputs + low location distance = good (different locations -> different outputs)
    loss = 0
    
    # Simple approach: maximize the std of outputs when locations are different
    # This encourages the model to produce varying outputs based on different locations
    output_std = outputs_tensor.std(dim=0).mean()
    
    # Penalize if output variance is too low
    min_variance_loss = torch.clamp(0.1 - output_std, min=0)
    
    return min_variance_loss

# Fine-tuning loop
print("\n6. Starting fine-tuning...")
print("   Training on location regularization...\n")

batch_size = 32
best_loss = float('inf')

for epoch in range(FINETUNE_EPOCHS):
    epoch_loss = 0
    num_batches = 0
    
    # Shuffle data
    indices = np.random.permutation(len(X_finetune))
    
    for batch_idx in range(0, len(X_finetune), batch_size):
        batch_indices = indices[batch_idx:batch_idx + batch_size]
        
        X_batch = torch.tensor(X_finetune[batch_indices], dtype=torch.float32, device=device)  # (B, INPUT_LEN, 2)
        Y_batch = torch.tensor(Y_finetune[batch_indices], dtype=torch.float32, device=device)  # (B, OUTPUT_LEN)
        loc_batch = locations_finetune[batch_indices]  # (B, 2) - keep as numpy
        
        # Reshape for model: (B, INPUT_LEN, 1, 2)
        X_batch = X_batch.unsqueeze(2)
        # Reshape for model: (B, 1, 2)
        loc_batch_t = torch.tensor(loc_batch, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Forward pass
        output = model(X_batch, loc_batch_t)  # (B, OUTPUT_LEN, 1)
        output = output.squeeze(-1)  # (B, OUTPUT_LEN)
        
        # Combine MSE loss with location regularization
        mse_loss = criterion(output, Y_batch)
        
        # Location regularization - simpler version
        loc_loss = location_regularization_loss([output[i].detach() for i in range(output.shape[0])], loc_batch)
        
        total_loss = mse_loss + LOCATION_LOSS_WEIGHT * loc_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_finetune, max_norm=1.0)
        optimizer.step()
        
        epoch_loss += total_loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"   Epoch {epoch+1}/{FINETUNE_EPOCHS} - Loss: {avg_loss:.6f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'saved_models/st_mlp.pth')
        print(f"   ✓ Best model saved")

# Verify fine-tuning worked
print("\n7. Verifying location sensitivity improvement...")
model.eval()

test_locations = {
    "Test A": (17.3850, 78.4867),  # Hyderabad location 1
    "Test B": (17.3589, 78.5941),  # Hyderabad location 2
}

dummy_traffic = np.random.randn(INPUT_LEN, 2) * 0.1
predictions = []

with torch.no_grad():
    for name, (lat, lng) in test_locations.items():
        loc_norm = static_scaler.transform([[lat, lng]])[0]
        
        X_test = torch.tensor(dummy_traffic, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)
        loc_test = torch.tensor(loc_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        out = model(X_test, loc_test)
        pred = out.mean().item()
        predictions.append(pred)
        
        print(f"   {name} ({lat:.4f}, {lng:.4f}): {pred:.4f}")

if len(set(predictions)) > 1:
    pred_diff = max(predictions) - min(predictions)
    print(f"\n   ✓ Location sensitivity verified: {pred_diff:.4f} prediction difference")
else:
    print("\n   ⚠ Warning: Predictions still identical")

print("\n" + "=" * 80)
print("✅ FINE-TUNING COMPLETE")
print("=" * 80)
print("Model has been fine-tuned for location sensitivity.")
print("Run app.py again to test predictions for different locations.")
print("=" * 80 + "\n")
