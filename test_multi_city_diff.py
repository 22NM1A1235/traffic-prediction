import sys
import numpy as np
import pandas as pd
from app import init_resources, predict_traffic_flow

print("="*80)
print("MULTI-CITY LOCATION DIFFERENTIATION TEST")
print("="*80)
print()

# Initialize app resources
init_resources()

# Test coordinates
test_locations = [
    {
        'name': 'Bangalore - Location 1',
        'lat': 12.9500,  # Bangalore area
        'lon': 77.5500,
        'city': 'Bangalore'
    },
    {
        'name': 'Bangalore - Location 2 (nearby)',
        'lat': 12.9502,  # Just 0.0002° away from Location 1
        'lon': 77.5502,
        'city': 'Bangalore'
    },
    {
        'name': 'Hyderabad - Location 1',
        'lat': 17.3850,  # Hyderabad center
        'lon': 78.4867,
        'city': 'Hyderabad'
    },
    {
        'name': 'Hyderabad - Location 2',
        'lat': 17.3900,  # Slightly different
        'lon': 78.4950,
        'city': 'Hyderabad'
    },
]

print("TEST SETUP:")
print(f"{'Location':<35} {'Latitude':>12} {'Longitude':>12} {'City':<12}")
print("-"*75)
for loc in test_locations:
    print(f"{loc['name']:<35} {loc['lat']:>12.4f} {loc['lon']:>12.4f} {loc['city']:<12}")
print()

# Get predictions for each location
predictions = {}
for loc in test_locations:
    try:
        pred = predict_traffic_flow(lat=loc['lat'], lng=loc['lon'])
        if pred and 'predictions' in pred:
            predictions[loc['name']] = np.array(pred['predictions'])
            print(f"✓ Got predictions for {loc['name']}")
        else:
            print(f"✗ Failed to get predictions for {loc['name']}: {pred}")
            predictions[loc['name']] = None
    except Exception as e:
        print(f"✗ Error predicting for {loc['name']}: {e}")
        predictions[loc['name']] = None

print()
print("="*80)
print("PREDICTIONS ANALYSIS")
print("="*80)
print()

# Display predictions for each location
for loc in test_locations:
    pred = predictions.get(loc['name'])
    if pred is not None:
        print(f"{loc['name']} ({loc['city']}):")
        print(f"  Mean: {pred.mean():.2f} veh/hr")
        print(f"  Min:  {pred.min():.2f}, Max: {pred.max():.2f}")
        print(f"  Predictions: {[f'{p:.1f}' for p in pred[:6]]}...(hours 1-6)")
        print()

# Compare within city vs across cities
print("="*80)
print("DIFFERENTIATION ANALYSIS")
print("="*80)
print()

# Within Bangalore
b1 = predictions.get('Bangalore - Location 1')
b2 = predictions.get('Bangalore - Location 2 (nearby)')
if b1 is not None and b2 is not None:
    mean_diff_bc = np.abs(b1.mean() - b2.mean())
    print(f"Within Bangalore (0.0002° apart):")
    print(f"  Mean difference: {mean_diff_bc:.3f} veh/hr")
    print(f"  Status: {'✓ GOOD' if mean_diff_bc > 0.2 else '✗ TOO SMALL'}")
    print()

# Within Hyderabad
h1 = predictions.get('Hyderabad - Location 1')
h2 = predictions.get('Hyderabad - Location 2')
if h1 is not None and h2 is not None:
    mean_diff_hy = np.abs(h1.mean() - h2.mean())
    print(f"Within Hyderabad (~5-10km apart):")
    print(f"  Mean difference: {mean_diff_hy:.3f} veh/hr")
    print(f"  Status: {'✓ GOOD' if mean_diff_hy > 0.3 else '✗ TOO SMALL'}")
    print()

# Across cities (THIS SHOULD BE HUGE!)
if b1 is not None and h1 is not None:
    mean_diff_city = np.abs(b1.mean() - h1.mean())
    max_diff_city = np.max(np.abs(b1 - h1))
    print(f"Between Cities (Bangalore vs Hyderabad - ~1500km apart):")
    print(f"  Mean difference: {mean_diff_city:.3f} veh/hr")
    print(f"  Max difference: {max_diff_city:.3f} veh/hr")
    print(f"  Status: {'✓✓✓ EXCELLENT' if mean_diff_city > 1.5 else ('✓✓ VERY GOOD' if mean_diff_city > 1.0 else ('✓ GOOD' if mean_diff_city > 0.5 else '✗ TOO SMALL'))}")
    print()

print("="*80)
print("EXPECTED BEHAVIOR:")
print("="*80)
print("✓ Bangalore & Hyderabad predictions should differ by 2+ veh/hr (CITY-LEVEL)")
print("✓ Locations within same city should differ by 0.3-0.5 veh/hr (FINER LOCATION)")
print("✗ Same coordinates should always give same prediction (DETERMINISTIC)")
