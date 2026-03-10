"""
Diagnose why different locations produce the same predictions
"""
import torch
import numpy as np
import pandas as pd
import pickle
from model import STMLP

# Load resources
print("=" * 60)
print("LOADING RESOURCES")
print("=" * 60)

with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/static_scaler.pkl', 'rb') as f:
    static_scaler = pickle.load(f)
with open('saved_models/sensor_ids.pkl', 'rb') as f:
    sensor_ids = pickle.load(f)

model = STMLP(1, 12, 2, 2, 64, 12, num_layers=3)
model.load_state_dict(torch.load('saved_models/st_mlp.pth'))
model.eval()

print(f"✓ Model loaded with 1 node")
print(f"✓ Scaler shape: {scaler.mean_.shape}")
print(f"✓ Static scaler mean: {static_scaler.mean_}")
print(f"✓ Static scaler scale: {static_scaler.scale_}")

# Load traffic data
df = pd.read_csv('traffic_time_series.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
pivot_flow = df.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()

scaler_mean = scaler.mean_
scaler_scale = scaler.scale_

# Test with two different locations in Hyderabad but same state
print("\n" + "=" * 60)
print("TEST 1: Same temporal data, different locations")
print("=" * 60)

# Use a sensor from the training set
test_sensor = sensor_ids[0]
print(f"\nUsing sensor: {test_sensor}")

# Get its historical data
if test_sensor in pivot_flow.columns:
    sensor_idx_scaler = sensor_ids.index(test_sensor)
    sensor_idx_flow = list(pivot_flow.columns).index(test_sensor)
    
    last_window_flow = pivot_flow.values[-12:, sensor_idx_flow]
    hours_norm = pivot_flow.index[-12:].hour.values / 23.0
    
    print(f"  Flow values: min={last_window_flow.min():.2f}, max={last_window_flow.max():.2f}")
    print(f"  Sensor index in scaler: {sensor_idx_scaler}")
    
    # Normalize
    last_window_flow_norm = (last_window_flow - scaler_mean[sensor_idx_scaler]) / scaler_scale[sensor_idx_scaler]
    
    # Input tensor
    input_combined = np.stack([last_window_flow_norm, hours_norm], axis=-1)
    input_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    
    # Test two different locations
    locations = [
        {"name": "Downtown Hyderabad", "lat": 17.3850, "lng": 78.4867},
        {"name": "Uptown Hyderabad", "lat": 17.4500, "lng": 78.5200},
        {"name": "Far East Hyderabad", "lat": 17.5000, "lng": 78.6000},
    ]
    
    print("\n--- Testing different locations ---\n")
    
    with torch.no_grad():
        for loc in locations:
            query_coords = np.array([[loc['lat'], loc['lng']]])
            query_coords_norm = static_scaler.transform(query_coords)[0]
            
            batch_static = torch.tensor(query_coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            out = model(input_tensor, batch_static)
            out_vals_norm = out.squeeze(0).squeeze(-1).numpy()
            out_vals = out_vals_norm * scaler_scale[sensor_idx_scaler] + scaler_mean[sensor_idx_scaler]
            
            print(f"{loc['name']:25s} ({loc['lat']:.4f}, {loc['lng']:.4f})")
            print(f"  Normalized coords: ({query_coords_norm[0]:.6f}, {query_coords_norm[1]:.6f})")
            print(f"  Raw coord diff from first: ({query_coords[0,0]-locations[0]['lat']:.4f}, {query_coords[0,1]-locations[0]['lng']:.4f})")
            print(f"  Normalized diff: ({query_coords_norm[0]-static_scaler.transform(np.array([[locations[0]['lat'], locations[0]['lng']]]))[0,0]:.6f}, {query_coords_norm[1]-static_scaler.transform(np.array([[locations[0]['lat'], locations[0]['lng']]]))[0,1]:.6f})")
            print(f"  Predictions (first 5): {out_vals[:5]}")
            print(f"  Prediction range: min={out_vals.min():.2f}, max={out_vals.max():.2f}, mean={out_vals.mean():.2f}")
            print()

# Test coordinate range
print("\n" + "=" * 60)
print("COORDINATE RANGE ANALYSIS")
print("=" * 60)

# Check the coordinate ranges in the training data
df_sensors = pd.read_csv('sensors.csv')
df_sensors_trained = df_sensors[df_sensors['sensor_id'].isin(sensor_ids)]

lat_range = df_sensors_trained['latitude'].max() - df_sensors_trained['latitude'].min()
lng_range = df_sensors_trained['longitude'].max() - df_sensors_trained['longitude'].min()

print(f"\nTraining data coordinate ranges:")
print(f"  Latitude:  {df_sensors_trained['latitude'].min():.4f} to {df_sensors_trained['latitude'].max():.4f} (range: {lat_range:.4f})")
print(f"  Longitude: {df_sensors_trained['longitude'].min():.4f} to {df_sensors_trained['longitude'].max():.4f} (range: {lng_range:.4f})")

print(f"\nStatic scaler parameters:")
print(f"  Mean: {static_scaler.mean_}")
print(f"  Scale (std): {static_scaler.scale_}")
print(f"  Inverse transform coefficient (1/scale): {1/static_scaler.scale_}")

# Check if normalized differences are too small
test_coords = [
    (17.3850, 78.4867),
    (17.3900, 78.4900),
    (17.3950, 78.4933),
]

print("\n\nNormalized differences for close locations:")
norms = [static_scaler.transform([c])[0] for c in test_coords]
print(f"  Coords 1: {test_coords[0]} -> Norm: {norms[0]}")
print(f"  Coords 2: {test_coords[1]} -> Norm: {norms[1]}")
print(f"  Coords 3: {test_coords[2]} -> Norm: {norms[2]}")
print(f"\n  Diff 1-2: raw=({test_coords[1][0]-test_coords[0][0]:.6f}, {test_coords[1][1]-test_coords[0][1]:.6f})")
print(f"           norm=({norms[1][0]-norms[0][0]:.6f}, {norms[1][1]-norms[0][1]:.6f})")
print(f"  Diff 2-3: raw=({test_coords[2][0]-test_coords[1][0]:.6f}, {test_coords[2][1]-test_coords[1][1]:.6f})")
print(f"           norm=({norms[2][0]-norms[1][0]:.6f}, {norms[2][1]-norms[1][1]:.6f})")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
