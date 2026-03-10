"""
Direct test of model predictions for different coordinates
"""
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from model import STMLP

print("\n" + "="*80)
print("DIRECT MODEL TEST - Different Coordinates")
print("="*80 + "\n")

# Load resources
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/static_scaler.pkl', 'rb') as f:
    static_scaler = pickle.load(f)
with open('saved_models/sensor_ids.pkl', 'rb') as f:
    sensor_ids = pickle.load(f)

print(f"Scaler: {len(scaler.mean_)} sensors")
print(f"Sensor IDs loaded: {len(sensor_ids)}")
print(f"Static scaler coefficients shape: {static_scaler.mean_.shape}")

# Load model
model = STMLP(1, 12, 2, 2, 64, 12, num_layers=2)
model.load_state_dict(torch.load('saved_models/st_mlp.pth'))
model.eval()
print(f"Model loaded and set to eval mode")

# Load traffic data
df_ts = pd.read_csv('traffic_time_series.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0

pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()

# Get a sensor's latest data
sensor_id = sensor_ids[0]
if sensor_id in pivot_flow.columns:
    flow_data = pivot_flow[sensor_id].values[-12:]
    hour_data = pivot_hour[sensor_id].values[-12:]
    
    sensor_idx = sensor_ids.index(sensor_id)
    flow_norm = (flow_data - scaler.mean_[sensor_idx]) / scaler.scale_[sensor_idx]
    
    print(f"\nUsing sensor: {sensor_id} (index {sensor_idx})")
    print(f"Flow data: min={flow_data.min():.2f}, max={flow_data.max():.2f}")
    print(f"Hour data: {hour_data[0]:.3f} to {hour_data[-1]:.3f}")
    
    # Test with different coordinates
    test_coords = [
        ("Cyberabad", 17.4400, 78.6100),
        ("Kukatpally", 17.3821, 78.4184),
        ("HITEC City", 17.3532, 78.4990),
    ]
    
    print(f"\nNormalizer ranges:")
    print(f"  Latitude: {static_scaler.mean_[0]:.4f} +/- {static_scaler.scale_[0]:.4f}")
    print(f"  Longitude: {static_scaler.mean_[1]:.4f} +/- {static_scaler.scale_[1]:.4f}")
    
    print(f"\nTesting model with different coordinates:\n")
    
    results = {}
    for loc_name, lat, lon in test_coords:
        # Normalize coordinates EXACTLY as Flask does
        coords_raw = np.array([[lat, lon]])
        coords_norm = static_scaler.transform(coords_raw)[0]
        
        # Create input
        input_combined = np.stack([flow_norm, hour_data], axis=-1)  # (12, 2)
        x_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1, 12, 1, 2)
        s_tensor = torch.tensor(coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # (1, 1, 2)
        
        # Predict
        with torch.no_grad():
            out = model(x_tensor, s_tensor)  # (1, 12, 1)
        
        # Denormalize
        pred_norm = out.squeeze().numpy()
        pred_denorm = pred_norm * scaler.scale_[sensor_idx] + scaler.mean_[sensor_idx]
        
        results[loc_name] = pred_denorm.mean()
        
        print(f"{loc_name:20} ({lat:.4f}, {lon:.4f})")
        print(f"  Normalized coords: {coords_norm}")
        print(f"  Mean prediction: {pred_denorm.mean():.4f}")
        print(f"  Min/Max: {pred_denorm.min():.4f} / {pred_denorm.max():.4f}")
        print()
    
    # Analysis
    means = list(results.values())
    print("="*80)
    print(f"Range: {max(means) - min(means):.4f}")
    print(f"Variation: {(max(means) - min(means)) / np.mean(means) * 100:.2f}%")
    
    if max(means) - min(means) > 0.1:
        print("PASS: Different locations produce different predictions!")
    else:
        print("FAIL: Predictions too similar!")
        
else:
    print(f"ERROR: Sensor {sensor_id} not in pivot_flow")

print()
