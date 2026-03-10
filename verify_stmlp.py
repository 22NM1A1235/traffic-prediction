"""
Verify STMLP Hyderabad model produces location-sensitive predictions
"""
import torch
import numpy as np
import pandas as pd
import pickle
from model import STMLP

print("\n" + "=" * 80)
print("VERIFYING STMLP HYDERABAD MODEL")
print("=" * 80 + "\n")

# Load model and resources
print("[1] Loading model and resources...")

# Load scalers and sensors
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/static_scaler.pkl', 'rb') as f:
    static_scaler = pickle.load(f)
with open('saved_models/sensor_ids.pkl', 'rb') as f:
    sensor_ids = pickle.load(f)

# Load model
model = STMLP(1, 12, 2, 2, 64, 12, num_layers=2)
model.load_state_dict(torch.load('saved_models/st_mlp.pth'))
model.eval()

print(f"   Model loaded")
print(f"   Sensors: {len(sensor_ids)}")

# Get test locations
print("\n[2] Getting test locations...")
df_sensors = pd.read_csv('sensors.csv')

test_locs = {}

# Banjara Hills
bh = df_sensors[(df_sensors['latitude'] > 17.36) & (df_sensors['latitude'] < 17.37) & 
                 (df_sensors['longitude'] > 78.46) & (df_sensors['longitude'] < 78.47)]
if len(bh) > 0:
    test_locs['Banjara Hills'] = [bh.iloc[0]['latitude'], bh.iloc[0]['longitude']]
    print(f"   Banjara Hills: {test_locs['Banjara Hills']}")

# Gachibowli  
gb = df_sensors[(df_sensors['latitude'] > 17.43) & (df_sensors['latitude'] < 17.44) & 
                 (df_sensors['longitude'] > 78.53) & (df_sensors['longitude'] < 78.54)]
if len(gb) > 0:
    test_locs['Gachibowli'] = [gb.iloc[0]['latitude'], gb.iloc[0]['longitude']]
    print(f"   Gachibowli: {test_locs['Gachibowli']}")

# Kukatpally
kp = df_sensors[(df_sensors['latitude'] > 17.38) & (df_sensors['latitude'] < 17.39) & 
                 (df_sensors['longitude'] > 78.41) & (df_sensors['longitude'] < 78.42)]
if len(kp) > 0:
    test_locs['Kukatpally'] = [kp.iloc[0]['latitude'], kp.iloc[0]['longitude']]
    print(f"   Kukatpally: {test_locs['Kukatpally']}")

# HITEC City
hc = df_sensors[(df_sensors['latitude'] > 17.35) & (df_sensors['latitude'] < 17.36) & 
                 (df_sensors['longitude'] > 78.49) & (df_sensors['longitude'] < 78.50)]
if len(hc) > 0:
    test_locs['HITEC City'] = [hc.iloc[0]['latitude'], hc.iloc[0]['longitude']]
    print(f"   HITEC City: {test_locs['HITEC City']}")

# Load traffic data
print("\n[3] Loading traffic data...")
df_ts = pd.read_csv('traffic_time_series.csv')
df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0

pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()

# Use latest data
flow_latest = pivot_flow.iloc[-12:][sensor_ids[0]].values
hour_latest = pivot_hour.iloc[-12:][sensor_ids[0]].values

# Normalize
flow_norm = (flow_latest - scaler.mean_[0]) / scaler.scale_[0]

print(f"   Latest flow: min={flow_latest.min():.2f}, max={flow_latest.max():.2f}")
print(f"   Hour range: {hour_latest[0]:.2f} to {hour_latest[-1]:.2f}")

# Test predictions
print("\n[4] Testing predictions for different locations...\n")
results = {}

for loc_name, [lat, lon] in test_locs.items():
    # Normalize coordinates
    coords = static_scaler.transform([[lat, lon]])[0]
    
    # Create input
    x_input = np.stack([flow_norm, hour_latest], axis=-1)  # (12, 2)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1, 12, 1, 2)
    s_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # (1, 1, 2)
    
    # Predict
    with torch.no_grad():
        out = model(x_tensor, s_tensor)  # (1, 12, 1)
    
    # Extract predictions and denormalize
    pred_norm = out.squeeze().numpy()
    pred_denorm = pred_norm * scaler.scale_[0] + scaler.mean_[0]
    
    results[loc_name] = {
        'norm': pred_norm,
        'denorm': pred_denorm,
        'mean': pred_denorm.mean()
    }
    
    print(f"{loc_name:20} -> Mean flow: {pred_denorm.mean():.4f}")

# Statistics
print("\n" + "=" * 80)
values = [r['mean'] for r in results.values()]
min_val =  min(values)
max_val = max(values)
range_val = max_val - min_val
var_pct = (range_val / np.mean(values)) * 100 if np.mean(values) != 0 else 0

print(f"Min prediction: {min_val:.4f}")
print(f"Max prediction: {max_val:.4f}")
print(f"Range: {range_val:.4f} ({var_pct:.2f}% variation)")
print("=" * 80)

if range_val > 0.1:
    print("✓ SUCCESS: Strong location sensitivity detected!")
    print("  Different coordinates produce meaningfully different predictions.")
else:
    print(f"⚠ Moderate sensitivity: {range_val:.4f} range ({var_pct:.2f}% variation)")

print()
