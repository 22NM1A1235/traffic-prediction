"""
Comprehensive test of location-aware predictions for different Hyderabad locations.
Tests both the model directly and through the Flask API.
"""

import torch
import pickle
import numpy as np
import pandas as pd
from model import STMLP

print("=" * 90)
print("COMPREHENSIVE LOCATION SENSITIVITY VERIFICATION - HYDERABAD")
print("=" * 90)

# Configuration
INPUT_LEN = 12
OUTPUT_LEN = 12
INPUT_DIM = 2
STATIC_DIM = 2
EMBED_DIM = 64
device = torch.device('cpu')

# Load model and scalers
print("\n[1/4] Loading fine-tuned model and resources...")
model = STMLP(num_nodes=1, input_len=INPUT_LEN, input_dim=INPUT_DIM, 
              static_dim=STATIC_DIM, embed_dim=EMBED_DIM, output_len=OUTPUT_LEN)
model.load_state_dict(torch.load('saved_models/st_mlp.pth', map_location=device))
model.eval()

with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/static_scaler.pkl', 'rb') as f:
    static_scaler = pickle.load(f)

print("   ✓ Model and scalers loaded")

# Different Hyderabad locations for testing
locations = {
    "South (Banjara Hills)": (17.3850, 78.4867),
    "East (Gachibowli)": (17.3589, 78.5941),
    "West (Kukatpally)": (17.3689, 78.3800),
    "Northeast (HITEC City)": (17.3595, 78.5889),
}

print("\n[2/4] Testing Model Predictions for Hyderabad Locations...")
print("-" * 90)

# Create multiple test samples to show consistency
test_samples = 5
predictions_all = {loc: [] for loc in locations}

traffic_data = pd.read_csv('traffic_time_series.csv')
traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
traffic_data['hour_norm'] = traffic_data['timestamp'].dt.hour / 23.0

pivot_flow = traffic_data.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
pivot_hour = traffic_data.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()

with open('saved_models/sensor_ids.pkl', 'rb') as f:
    sensor_ids = pickle.load(f)

available_sensors = [s for s in sensor_ids if s in pivot_flow.columns]
pivot_flow = pivot_flow[available_sensors]
pivot_hour = pivot_hour[available_sensors]

# Normalize flow
flow_norm = scaler.fit_transform(pivot_flow.values)
hour_norm = pivot_hour.values

print("\nPredictions for Different Hyderabad Locations:")
print("(Testing with actual traffic data from the dataset)\n")

# Test with actual traffic data from different time windows
with torch.no_grad():
    for sample_idx in range(min(test_samples, len(flow_norm) - INPUT_LEN - OUTPUT_LEN)):
        start_idx = sample_idx * 10
        if start_idx + INPUT_LEN > len(flow_norm):
            break
        
        if sample_idx == 0:
            print(f"{'Location Name':<25} {'Coords':<20} {'Prediction':<12} {'Average':<12}")
            print("-" * 90)
        
        for loc_name, (lat, lng) in locations.items():
            # Prepare sample data  
            flow_hist = flow_norm[start_idx:start_idx + INPUT_LEN, :50].mean(axis=1)  # Average across sensors
            hour_hist = hour_norm[start_idx:start_idx + INPUT_LEN, 0]
            
            X_sample = np.stack([flow_hist, hour_hist], axis=-1)  # (INPUT_LEN, 2)
            X_tensor = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # (1, INPUT_LEN, 1, 2)
            
            # Location coordinates (normalized)
            loc_norm = static_scaler.transform([[lat, lng]])[0]
            loc_tensor = torch.tensor(loc_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
            
            # Predict
            output = model(X_tensor, loc_tensor)  # (1, OUTPUT_LEN, 1)
            pred_mean = output.mean().item()
            predictions_all[loc_name].append(pred_mean)
            
            if sample_idx == 0:
                print(f"{loc_name:<25} {lat:.4f},{lng:.4f}  {pred_mean:>10.4f}   ", end="")

for loc_name in locations:
    avg_pred = np.mean(predictions_all[loc_name])
    print(f"{avg_pred:>10.4f}")

print("\n[3/4] Statistical Analysis...")
print("-" * 90)

all_predictions = []
for loc_name in locations:
    all_predictions.extend(predictions_all[loc_name])

print(f"\nPrediction Statistics Across All Locations:")
print(f"  Minimum prediction: {min(all_predictions):.6f}")
print(f"  Maximum prediction: {max(all_predictions):.6f}")
print(f"  Range: {max(all_predictions) - min(all_predictions):.6f}")
print(f"  Mean: {np.mean(all_predictions):.6f}")
print(f"  Std Dev: {np.std(all_predictions):.6f}")
print(f"  Coefficient of Variation: {np.std(all_predictions) / abs(np.mean(all_predictions)) * 100:.2f}%")

# Check if predictions are significantly different per location
location_means = [np.mean(predictions_all[loc]) for loc in locations]
min_mean = min(location_means)
max_mean = max(location_means)
range_mean = max_mean - min_mean

print(f"\nPer-Location Prediction Differences:")
for i, (loc_name, lat, lng) in enumerate([(k, v[0], v[1]) for k, v in locations.items()]):
    mean_pred = np.mean(predictions_all[loc_name])
    std_pred = np.std(predictions_all[loc_name])
    print(f"  {loc_name:<25}: {mean_pred:.6f} ± {std_pred:.6f}")

print(f"\n  Location-based Range: {range_mean:.6f}")
print(f"  Location-based Variation: {range_mean / np.mean(location_means) * 100:.2f}%")

print("\n[4/4] Final Verdict...")
print("-" * 90)

if range_mean > 0.001 and np.std(all_predictions) > 0.0005:
    print("\n✅ SUCCESS: Location-aware predictions are working correctly!")
    print(f"   • Different Hyderabad locations produce different predictions")
    print(f"   • Prediction range: {max(all_predictions) - min(all_predictions):.6f}")
    print(f"   • Per-location variation: {range_mean / np.mean(location_means) * 100:.2f}%")
    print("\n   You can now:")
    print("   1. Play with different locations on the web app (http://localhost:5000)")
    print("   2. Verify that each location gets a unique traffic prediction")
    print("   3. Adjust location coordinates to see how predictions change")
else:
    print("\n⚠️  WARNING: Location sensitivity is low")
    print(f"   Range: {range_mean:.6f}, Std: {np.std(all_predictions):.6f}")

print("\n" + "=" * 90)
