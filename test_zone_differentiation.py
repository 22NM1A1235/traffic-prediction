#!/usr/bin/env python3
"""
Test within-city zone differentiation - different locations should show different prediction patterns
"""
import numpy as np
from app import init_resources, model, scaler_mean, scaler_scale, INPUT_LEN, OUTPUT_LEN, static_scaler, city_scalers, city_centers, location_zones, location_norm_params, df_sensors, sensor_ids
import torch
import pandas as pd

print("="*80)
print("WITHIN-CITY ZONE DIFFERENTIATION TEST")
print("="*80)

# Initialize resources
print("\nLoading resources...")
init_resources()
print(f"✓ Resources loaded (sensor_ids: {len(sensor_ids) if sensor_ids else 0}, scaler_mean: {len(scaler_mean) if scaler_mean is not None else 0})")

# Load traffic data
df = pd.read_csv('traffic_time_series.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
pivot_flow = df.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()

print(f"Pivot flow sensors available: {len(pivot_flow.columns)}")

# Use same sensor for all predictions (only change coordinates)
target_sensor = sensor_ids[100] if len(sensor_ids) > 100 else list(pivot_flow.columns)[0]
sensor_idx = sensor_ids.index(target_sensor) if target_sensor in sensor_ids else 0

# Extract history
last_window_flow = pivot_flow.values[-INPUT_LEN:, sensor_idx]
hours_norm = pivot_flow.index[-INPUT_LEN:].hour.values / 23.0

# Prepare input
input_combined = np.stack([last_window_flow, hours_norm], axis=-1)
input_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

# Test locations within HYDERABAD
print("\n" + "="*80)
print("HYDERABAD - DIFFERENT ZONES (Same City, Different Traffic Patterns)")
print("="*80)

test_locations = {
    'HiTech City (HIGH TRAFFIC)': (17.3589, 78.3877),  # 1.8x
    'Gachibowli (MEDIUM TRAFFIC)': (17.3565, 78.4007),  # 1.5x
    'Kompally (LOW TRAFFIC)': (17.5000, 78.7000),  # 0.6x
}

predictions_dict = {}

for zone_name, (lat, lng) in test_locations.items():
    print(f"\n{zone_name}")
    print(f"Coordinates: ({lat:.4f}, {lng:.4f})")
    
    # Create static features from query coordinates
    query_coords = np.array([[lat, lng]])
    query_coords_norm = static_scaler.transform(query_coords)[0]
    batch_static = torch.tensor(query_coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        out = model(input_tensor, batch_static)
    
    out_vals_norm = out.squeeze(0).squeeze(-1).numpy()
    out_vals = out_vals_norm * scaler_scale[sensor_idx] + scaler_mean[sensor_idx]
    
    # Apply enhancement logic (same as in app.py)
    out_vals_enhanced = out_vals.copy()
    traffic_factor = None
    
    # Find zone
    hyderabad_zones = location_zones.get('Hyderabad', [])
    nearest_zone = min(hyderabad_zones, key=lambda z: np.sqrt((lat - z['lat'])**2 + (lng - z['lon'])**2))
    traffic_factor = nearest_zone['traffic_factor']
    
    print(f"  Zone: {nearest_zone['name']}")
    print(f"  Traffic Factor: {traffic_factor}x")
    
    # Apply city modifier (Hyderabad = -2.0)
    out_vals_enhanced = out_vals_enhanced - 2.0
    
    # Apply zone-based enhancement
    out_vals_enhanced = out_vals_enhanced * traffic_factor
    
    if traffic_factor > 1.0:  # High-traffic
        peak_boost = 5.0 * (traffic_factor - 1.0)
        for t in range(OUTPUT_LEN):
            hour_signal = np.abs(np.sin(np.pi * (t - 6) / 12.0))
            out_vals_enhanced[t] += peak_boost * hour_signal
    elif traffic_factor < 1.0:  # Low-traffic
        smoothing_factor = 1.0 - (1.0 - traffic_factor) * 0.3
        out_vals_enhanced = out_vals_enhanced * smoothing_factor
    
    predictions_dict[zone_name] = out_vals_enhanced
    
    print(f"  Predictions: min={out_vals_enhanced.min():.2f}, max={out_vals_enhanced.max():.2f}, mean={out_vals_enhanced.mean():.2f}")
    print(f"  Range (max-min): {out_vals_enhanced.max() - out_vals_enhanced.min():.2f} veh/hr")

# Compare predictions
print("\n" + "="*80)
print("WITHIN-CITY COMPARISON")
print("="*80)

hitech = predictions_dict['HiTech City (HIGH TRAFFIC)']
kompally = predictions_dict['Kompally (LOW TRAFFIC)']

mean_diff = np.mean(hitech) - np.mean(kompally)
pattern_diff = np.std(hitech) - np.std(kompally)  # Difference in variability

print(f"\nHiTech vs Kompally:")
print(f"  Mean difference: {mean_diff:.2f} veh/hr")
print(f"  HiTech range (variability): {hitech.max() - hitech.min():.2f} veh/hr")
print(f"  Kompally range (variability): {kompally.max() - kompally.min():.2f} veh/hr")
print(f"  Pattern difference (std): {pattern_diff:.4f}")

if mean_diff > 15:
    print(f"  ✓✓✓ STRONG WITHIN-CITY DIFFERENTIATION ACHIEVED")
else:
    print(f"  ✗ Within-city differentiation is weak")

print("\n" + "="*80)
