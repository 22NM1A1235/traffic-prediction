#!/usr/bin/env python
"""
Test script to verify location-specific predictions
Tests that different query locations produce different traffic predictions
"""

import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from model import STMLP

# Configuration
INPUT_LEN = 12
OUTPUT_LEN = 12
INPUT_DIM = 2
STATIC_DIM = 2
EMBED_DIM = 64

def test_location_differentiation():
    """Test that different locations produce different predictions"""
    
    print("=" * 70)
    print("TESTING LOCATION-SPECIFIC PREDICTIONS")
    print("=" * 70)
    
    # Load saved resources
    print("\n1. Loading saved resources...")
    with open('saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('saved_models/static_scaler.pkl', 'rb') as f:
        static_scaler = pickle.load(f)
    with open('saved_models/sensor_ids.pkl', 'rb') as f:
        sensor_ids = pickle.load(f)
    
    print(f"   ✓ Loaded scaler with {len(scaler.mean_)} sensors")
    print(f"   ✓ Loaded static scaler")
    print(f"   ✓ Loaded {len(sensor_ids)} sensor IDs")
    
    # Load model
    print("\n2. Loading trained model...")
    model = STMLP(1, INPUT_LEN, INPUT_DIM, STATIC_DIM, EMBED_DIM, OUTPUT_LEN, num_layers=3)
    model.load_state_dict(torch.load('saved_models/st_mlp.pth'))
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Load traffic data
    print("\n3. Loading traffic data...")
    df_ts = pd.read_csv('traffic_time_series.csv')
    df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
    pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
    print(f"   ✓ Loaded {len(df_ts)} traffic records")
    print(f"   ✓ Pivot shape: {pivot_flow.shape}")
    
    # Get last 12 hours of data (for all sensors initially)
    last_window = pivot_flow.values[-INPUT_LEN:]  # (12, num_sensors)
    hours_norm = pivot_flow.index[-INPUT_LEN:].hour.values / 23.0
    
    print(f"   ✓ Using last {INPUT_LEN} time steps for input")
    
    # Test 3 different locations
    print("\n4. Testing predictions for different locations...")
    print("   " + "-" * 66)
    
    test_locations = [
        {"name": "Location 1", "lat": 12.9716, "lng": 77.5946},
        {"name": "Location 2", "lat": 12.9720, "lng": 77.5950},  # Nearby
        {"name": "Location 3", "lat": 13.0000, "lng": 77.6500},  # Different area
    ]
    
    predictions = []
    
    for loc_test in test_locations:
        lat, lng = loc_test['lat'], loc_test['lng']
        print(f"\n   Testing {loc_test['name']}: ({lat:.4f}, {lng:.4f})")
        
        try:
            # Find nearest sensor
            from sklearn.metrics.pairwise import euclidean_distances
            df_sensors = pd.read_csv('sensors.csv')
            df_sensors_filtered = df_sensors[df_sensors['sensor_id'].isin(sensor_ids)]
            distances = np.sqrt(
                (df_sensors_filtered['latitude'] - lat)**2 + 
                (df_sensors_filtered['longitude'] - lng)**2
            )
            nearest_idx = distances.idxmin()
            nearest_sensor = df_sensors_filtered.loc[nearest_idx]
            sensor_id = nearest_sensor['sensor_id']
            
            # Get sensor's flow data
            if sensor_id in pivot_flow.columns:
                sensor_idx_scaler = sensor_ids.index(sensor_id)
                last_window_sensor = pivot_flow[sensor_id].values[-INPUT_LEN:]
                
                # Normalize
                last_window_norm = (last_window_sensor - scaler.mean_[sensor_idx_scaler]) / scaler.scale_[sensor_idx_scaler]
                
                # Query coordinates
                query_coords = np.array([[lat, lng]])
                query_coords_norm = static_scaler.transform(query_coords)[0]
                
                # Prepare tensors
                input_combined = np.stack([last_window_norm, hours_norm], axis=-1)
                input_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                batch_static = torch.tensor(query_coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    out = model(input_tensor, batch_static)
                
                # Denormalize
                out_vals_norm = out.squeeze(0).squeeze(-1).numpy()
                out_vals = out_vals_norm * scaler.scale_[sensor_idx_scaler] + scaler.mean_[sensor_idx_scaler]
                
                predictions.append({
                    'location': loc_test['name'],
                    'coords': (lat, lng),
                    'sensor': sensor_id,
                    'predictions': out_vals,
                    'mean': out_vals.mean(),
                    'min': out_vals.min(),
                    'max': out_vals.max()
                })
                
                print(f"      Nearest Sensor: {sensor_id}")
                print(f"      Predictions Mean: {out_vals.mean():.4f}")
                print(f"      Predictions Range: [{out_vals.min():.4f}, {out_vals.max():.4f}]")
                print(f"      First 3 predictions: {out_vals[:3]}")
            else:
                print(f"      ✗ Sensor {sensor_id} not in traffic data")
        
        except Exception as e:
            print(f"      ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare predictions
    print("\n5. Comparison of predictions...")
    print("   " + "-" * 66)
    
    if len(predictions) >= 2:
        pred1 = predictions[0]
        pred2 = predictions[1]
        
        # Compare first predictions
        diff = np.abs(pred1['predictions'] - pred2['predictions'])
        print(f"\n   {pred1['location']} vs {pred2['location']}:")
        print(f"      Mean difference: {diff.mean():.4f}")
        print(f"      Max difference: {diff.max():.4f}")
        print(f"      All predictions identical? {np.allclose(pred1['predictions'], pred2['predictions'])}")
        
        if not np.allclose(pred1['predictions'], pred2['predictions']):
            print(f"      ✓ PASS: Different locations produce DIFFERENT predictions!")
        else:
            print(f"      ✗ FAIL: Different locations produce SAME predictions")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    test_location_differentiation()
