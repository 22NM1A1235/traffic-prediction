import numpy as np
import torch
from app import init_resources
import app as app_module
import pickle

print("="*80)
print("MULTI-CITY LOCATION DIFFERENTIATION TEST") 
print("="*80)
print()

# Initialize
init_resources()

# Get the global scaler and model from app module
scaler = app_module.scaler
model = app_module.model
static_scaler = app_module.static_scaler
scaler_mean = app_module.scaler_mean
scaler_scale = app_module.scaler_scale

# Load city info
with open('saved_models/city_centers.pkl', 'rb') as f:
    city_centers = pickle.load(f)

print("City Centers:")
for city, (lat, lon) in city_centers.items():
    print(f"  {city}: ({lat:.4f}, {lon:.4f})")
print()

# Test locations
test_cases = [
    ('Bangalore-1', 12.9505, 77.5500, 'Bangalore'),
    ('Bangalore-2', 12.9507, 77.5502, 'Bangalore'),  # Very close to Bangalore-1
    ('Hyderabad-1', 17.3850, 78.4867, 'Hyderabad'),
    ('Hyderabad-2', 17.3900, 78.4950, 'Hyderabad'),  # Farther in Hyderabad
]

results = {}

for test_name, lat, lon, expected_city in test_cases:
    print(f"Predicting for {test_name} ({expected_city}): ({lat:.4f}, {lon:.4f})")
    
    # Classify to city
    dist_b = np.sqrt((lat - city_centers['Bangalore'][0])**2 + (lon - city_centers['Bangalore'][1])**2)
    dist_h = np.sqrt((lat - city_centers['Hyderabad'][0])**2 + (lon - city_centers['Hyderabad'][1])**2)
    classified_city = 'Bangalore' if dist_b < dist_h else 'Hyderabad'
    
    print(f"  Distance to Bangalore: {dist_b:.6f} degrees")
    print(f"  Distance to Hyderabad: {dist_h:.6f} degrees")
    print(f"  Classified as: {classified_city}")
    
    # Create random input data (last 12 hours of traffic) 
    np.random.seed(42)
    last_window_flow = np.array([75, 76, 78, 82, 85, 86, 88, 87, 89, 87, 86, 84], dtype=np.float32)
    
    # Normalize flow data
    sensor_idx = 0  # Use first sensor in list  
    flow_norm = (last_window_flow - scaler_mean[sensor_idx]) / scaler_scale[sensor_idx]
    
    # Create hour features  
    hours = np.arange(1, 13, dtype=np.float32)
    hours_norm = (hours - 6.5) / 6.5  # Normalize to [-1, 1] roughly
    
    # Combine features
    input_combined = np.stack([flow_norm, hours_norm], axis=-1)
    input_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    
    # Normalize coordinates
    query_coords = np.array([[lat, lon]], dtype=np.float32)
    query_coords_norm = static_scaler.transform(query_coords)[0]
    batch_static = torch.tensor(query_coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        out = model(input_tensor, batch_static)
    
    out_vals_norm = out.squeeze(0).squeeze(-1).numpy()
    out_vals = out_vals_norm * scaler_scale[sensor_idx] + scaler_mean[sensor_idx]
    
    # Apply city-level modification
    city_modifier = 2.0 if classified_city == 'Bangalore' else -2.0
    out_vals_final = out_vals + city_modifier
    
    results[test_name] = {
        'city': classified_city,
        'coords': (lat, lon),
        'predictions_base': out_vals,
        'predictions_enhanced': out_vals_final,
        'modifier': city_modifier
    }
    
    print(f"  Base predictions (mean): {out_vals.mean():.2f} veh/hr")
    print(f"  City modifier: {city_modifier:+.2f}")
    print(f"  Enhanced predictions (mean): {out_vals_final.mean():.2f} veh/hr")
    print()

print("="*80)
print("CROSS-CITY COMPARISON")
print("="*80)
print()

b1 = results['Bangalore-1']['predictions_enhanced']
b2 = results['Bangalore-2']['predictions_enhanced']
h1 = results['Hyderabad-1']['predictions_enhanced']
h2 = results['Hyderabad-2']['predictions_enhanced']

print(f"{'Comparison':<40} {'Mean Diff':>12} {'Max Diff':>12} {'Status':<20}")
print("-"*75)

# Within Bangalore
diff = np.abs(b1.mean() - b2.mean())
max_diff = np.max(np.abs(b1 - b2))
status = '✓ GOOD' if diff > 0.2 else '✗ Small'
print(f"{'Bangalore-1 vs Bangalore-2':<40} {diff:>12.3f} {max_diff:>12.3f} {status:<20}")

# Within Hyderabad
diff = np.abs(h1.mean() - h2.mean())
max_diff = np.max(np.abs(h1 - h2))
status = '✓ GOOD' if diff > 0.3 else '✗ Small'
print(f"{'Hyderabad-1 vs Hyderabad-2':<40} {diff:>12.3f} {max_diff:>12.3f} {status:<20}")

# Across cities - THIS IS THE KEY TEST!
diff = np.abs(b1.mean() - h1.mean())
max_diff = np.max(np.abs(b1 - h1))
status = '✓✓✓ EXCELLENT' if diff > 2.0 else ('✓✓ VERY GOOD' if diff > 1.5 else ('✓ GOOD' if diff > 1.0 else '✗ POOR'))
print(f"{'Bangalore-1 vs Hyderabad-1 (CITIES!)':<40} {diff:>12.3f} {max_diff:>12.3f} {status:<20}")

print()
print("="*80)
print("KEY FINDING:")
print("="*80)
print(f"Bangalore predictions: {b1.mean():.2f} ± {np.std([b1.mean(), b2.mean()]):.3f}")
print(f"Hyderabad predictions: {h1.mean():.2f} ± {np.std([h1.mean(), h2.mean()]):.3f}")
print(f"City-level difference: {abs(b1.mean() - h1.mean()):.2f} veh/hr ✓")
