"""
Create a local coordinate scaler for Hyderabad region
This ensures fine-grained location differentiation within the city
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

print("Creating local coordinate scaler for Hyderabad...")

# Load sensors
df_sensors = pd.read_csv('sensors.csv')

# Define Hyderabad region boundaries (slightly wider than sensor coverage for safety)
# Trained sensors: Lat 12.9001 to 17.4189, Lon 77.5 to 78.5495
# We'll add a small buffer
HYDERABAD_LAT_MIN = 12.8
HYDERABAD_LAT_MAX = 17.6
HYDERABAD_LON_MIN = 77.4
HYDERABAD_LON_MAX = 78.7

# Filter to Hyderabad region
df_hyderabad = df_sensors[
    (df_sensors['latitude'] >= HYDERABAD_LAT_MIN) &
    (df_sensors['latitude'] <= HYDERABAD_LAT_MAX) &
    (df_sensors['longitude'] >= HYDERABAD_LON_MIN) &
    (df_sensors['longitude'] <= HYDERABAD_LON_MAX)
]

print(f"\nSensors in Hyderabad region: {len(df_hyderabad)} out of {len(df_sensors)}")
print(f"Lat range: {df_hyderabad['latitude'].min():.4f} to {df_hyderabad['latitude'].max():.4f}")
print(f"Lon range: {df_hyderabad['longitude'].min():.4f} to {df_hyderabad['longitude'].max():.4f}")

# Create a local scaler based ONLY on Hyderabad coordinates
hyderabad_coords = df_hyderabad[['latitude', 'longitude']].values

# Create the local scaler
local_static_scaler = StandardScaler()
local_static_scaler.fit(hyderabad_coords)

print(f"\nLocal scaler (Hyderabad only):")
print(f"  Mean: {local_static_scaler.mean_}")
print(f"  Scale: {local_static_scaler.scale_}")

# Save it with a new name
with open('saved_models/static_scaler_hyderabad_local.pkl', 'wb') as f:
    pickle.dump(local_static_scaler, f)

print("\n✓ Saved to saved_models/static_scaler_hyderabad_local.pkl")

# Test with nearby coordinates
print("\n" + "="*60)
print("TESTING LOCAL SCALER vs GLOBAL SCALER")
print("="*60)

# Load global scaler for comparison
with open('saved_models/static_scaler.pkl', 'rb') as f:
    global_scaler = pickle.load(f)

test_coords = [
    (17.3850, 78.4867),
    (17.3900, 78.4900),
    (17.3950, 78.4933),
]

print("\nNormalization comparison for three close locations:")
print("-" * 60)

global_norms = [global_scaler.transform([c])[0] for c in test_coords]
local_norms = [local_static_scaler.transform([c])[0] for c in test_coords]

for i, coord in enumerate(test_coords):
    print(f"\nLocation {i+1}: {coord}")
    print(f"  Global norm: {global_norms[i]} -> L2: {np.linalg.norm(global_norms[i]):.4f}")
    print(f"  Local norm:  {local_norms[i]} -> L2: {np.linalg.norm(local_norms[i]):.4f}")

print("\n" + "-" * 60)
print("Differences between point 1 and 2:")
print(f"  Global diff: {global_norms[1] - global_norms[0]} (L2: {np.linalg.norm(global_norms[1] - global_norms[0]):.6f})")
print(f"  Local diff:  {local_norms[1] - local_norms[0]} (L2: {np.linalg.norm(local_norms[1] - local_norms[0]):.6f})")

print("\nDifferences between point 2 and 3:")
print(f"  Global diff: {global_norms[2] - global_norms[1]} (L2: {np.linalg.norm(global_norms[2] - global_norms[1]):.6f})")
print(f"  Local diff:  {local_norms[2] - local_norms[1]} (L2: {np.linalg.norm(local_norms[2] - local_norms[1]):.6f})")

improvement_ratio = (np.linalg.norm(local_norms[1] - local_norms[0]) / 
                     np.linalg.norm(global_norms[1] - global_norms[0]))
print(f"\n✓ Local scaler provides {improvement_ratio:.1f}x better location differentiation!")
