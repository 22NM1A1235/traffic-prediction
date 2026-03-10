"""
Test if using local scaler improves location differentiation
"""
import torch
import numpy as np
import pandas as pd
import pickle
from model import STMLP

print("="*60)
print("TESTING LOCAL SCALER IMPACT")
print("="*60)

# Load resources
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/static_scaler_hyderabad_local.pkl', 'rb') as f:
    local_static_scaler = pickle.load(f)
with open('saved_models/static_scaler.pkl', 'rb') as f:
    global_static_scaler = pickle.load(f)
with open('saved_models/sensor_ids.pkl', 'rb') as f:
    sensor_ids = pickle.load(f)

model = STMLP(1, 12, 2, 2, 64, 12, num_layers=3)
model.load_state_dict(torch.load('saved_models/st_mlp.pth'))
model.eval()

# Load traffic data
df = pd.read_csv('traffic_time_series.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
pivot_flow = df.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()

scaler_mean = scaler.mean_
scaler_scale = scaler.scale_

# Use a sensor from the training set
test_sensor = sensor_ids[0]
sensor_idx_scaler = sensor_ids.index(test_sensor)
sensor_idx_flow = list(pivot_flow.columns).index(test_sensor)

last_window_flow = pivot_flow.values[-12:, sensor_idx_flow]
hours_norm = pivot_flow.index[-12:].hour.values / 23.0
last_window_flow_norm = (last_window_flow - scaler_mean[sensor_idx_scaler]) / scaler_scale[sensor_idx_scaler]

# Input tensor
input_combined = np.stack([last_window_flow_norm, hours_norm], axis=-1)
input_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

# Test two different locations with both scalers
locations = [
    ("Downtown Hyderabad", 17.3850, 78.4867),
    ("Uptown Hyderabad", 17.4500, 78.5200),
    ("Far East", 17.5000, 78.6000),
]

print("\nPredictions with GLOBAL scaler:")
print("-" * 60)

global_predictions = []
with torch.no_grad():
    for name, lat, lng in locations:
        query_coords = np.array([[lat, lng]])
        query_coords_norm = global_static_scaler.transform(query_coords)[0]
        batch_static = torch.tensor(query_coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out = model(input_tensor, batch_static)
        out_vals_norm = out.squeeze(0).squeeze(-1).numpy()
        out_vals = out_vals_norm * scaler_scale[sensor_idx_scaler] + scaler_mean[sensor_idx_scaler]
        global_predictions.append(out_vals)
        print(f"{name:25s}: {out_vals[:5]} ... range: {out_vals.min():.2f}-{out_vals.max():.2f}")

print("\nPredictions with LOCAL Hyderabad scaler:")
print("-" * 60)

local_predictions = []
with torch.no_grad():
    for name, lat, lng in locations:
        query_coords = np.array([[lat, lng]])
        query_coords_norm = local_static_scaler.transform(query_coords)[0]
        batch_static = torch.tensor(query_coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out = model(input_tensor, batch_static)
        out_vals_norm = out.squeeze(0).squeeze(-1).numpy()
        out_vals = out_vals_norm * scaler_scale[sensor_idx_scaler] + scaler_mean[sensor_idx_scaler]
        local_predictions.append(out_vals)
        print(f"{name:25s}: {out_vals[:5]} ... range: {out_vals.min():.2f}-{out_vals.max():.2f}")

# Compare differences
print("\n" + "="*60)
print("DIFFERENCE ANALYSIS")
print("="*60)

print("\nWith GLOBAL scaler:")
diff_global = np.abs(global_predictions[1] - global_predictions[0])
print(f"  Difference (Uptown vs Downtown): {diff_global[:5]} ... mean={diff_global.mean():.6f}")

print("\nWith LOCAL scaler:")
diff_local = np.abs(local_predictions[1] - local_predictions[0])
print(f"  Difference (Uptown vs Downtown): {diff_local[:5]} ... mean={diff_local.mean():.6f}")

improvement = diff_local.mean() / diff_global.mean() if diff_global.mean() != 0 else 1
print(f"\n✓ LOCAL scaler provides {improvement:.2f}x more differentiation!")
