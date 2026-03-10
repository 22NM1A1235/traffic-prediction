"""
Test Hyderabad model - verify different locations give different predictions
"""
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 80)
print("TESTING HYDERABAD-TRAINED MODEL")
print("=" * 80 + "\n")

# Load model
state = torch.load('saved_models/st_mlp.pth', weights_only=False)
print(f"[1] Model loaded: {len(state)} parameters")

# Load scalers and sensors
with open('saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('saved_models/static_scaler.pkl', 'rb') as f:
    static_scaler = pickle.load(f)
with open('saved_models/sensor_ids.pkl', 'rb') as f:
    sensor_ids = pickle.load(f)

print(f"[2] Loaded: {len(sensor_ids)} sensors, scalers ready")

# Get test locations
df_sensors = pd.read_csv('sensors.csv')
test_locs = {}

# Banjara Hills
bh = df_sensors[(df_sensors['latitude'] > 17.36) & (df_sensors['latitude'] < 17.37) & 
                 (df_sensors['longitude'] > 78.46) & (df_sensors['longitude'] < 78.47)]
if len(bh) > 0:
    test_locs['Banjara Hills'] = [bh.iloc[0]['latitude'], bh.iloc[0]['longitude']]

# Gachibowli  
gb = df_sensors[(df_sensors['latitude'] > 17.43) & (df_sensors['latitude'] < 17.44) & 
                 (df_sensors['longitude'] > 78.53) & (df_sensors['longitude'] < 78.54)]
if len(gb) > 0:
    test_locs['Gachibowli'] = [gb.iloc[0]['latitude'], gb.iloc[0]['longitude']]

# Kukatpally
kp = df_sensors[(df_sensors['latitude'] > 17.38) & (df_sensors['latitude'] < 17.39) & 
                 (df_sensors['longitude'] > 78.41) & (df_sensors['longitude'] < 78.42)]
if len(kp) > 0:
    test_locs['Kukatpally'] = [kp.iloc[0]['latitude'], kp.iloc[0]['longitude']]

# HITEC City
hc = df_sensors[(df_sensors['latitude'] > 17.35) & (df_sensors['latitude'] < 17.36) & 
                 (df_sensors['longitude'] > 78.49) & (df_sensors['longitude'] < 78.50)]
if len(hc) > 0:
    test_locs['HITEC City'] = [hc.iloc[0]['latitude'], hc.iloc[0]['longitude']]

print(f"[3] Test locations: {list(test_locs.keys())}\n")

# Create simple test data (same traffic pattern, different locations)
test_flow = np.ones((12, len(sensor_ids))) * 0.5  # Normalized flow
test_hour = np.ones((12, len(sensor_ids))) * (17 / 23.0)  # 5 PM

# Test predictions
results = {}

for loc_name, [lat, lon] in test_locs.items():
    # Normalize coordinates
    coords = static_scaler.transform([[lat, lon]])[0]
    
    # Create input
    x_input = np.stack([test_flow[:, 0], test_hour[:, 0]], axis=-1)  # shape (12, 2)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)  # (1, 12, 2)
    s_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # (1, 2)
    
    # Predict (using the simple model structure)
    with torch.no_grad():
        fc1_weight = list(state.values())[0]
        fc1_bias = list(state.values())[1]
        fc2_weight = list(state.values())[2]
        fc2_bias = list(state.values())[3]
        
        x_flat = x_tensor.reshape(1, -1)  # (1, 24)
        x_combined = torch.cat([x_flat, s_tensor], dim=1)  # (1, 26)
        
        hidden = torch.nn.functional.relu(torch.nn.functional.linear(x_combined, fc1_weight, fc1_bias))
        output = torch.nn.functional.linear(hidden, fc2_weight, fc2_bias)
        pred = output.squeeze().mean().item()
    
    results[loc_name] = pred
    print(f"{loc_name:20} -> {pred:.6f}")

# Statistics
if len(results) > 0:
    values = list(results.values())
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    var_pct = (range_val / np.mean(values)) * 100 if np.mean(values) != 0 else 0
    
    print("\n" + "=" * 80)
    print("RESULTS:")
    print(f"  Min prediction: {min_val:.6f}")
    print(f"  Max prediction: {max_val:.6f}")
    print(f"  Range: {range_val:.6f} ({var_pct:.2f}% variation)")
    print("=" * 80)
    
    if range_val > 0.005:
        print("✓ SUCCESS: Different locations produce different predictions!")
    else:
        print("✗ ISSUE: Predictions still too similar")
else:
    print("Could not find test locations")

print()
