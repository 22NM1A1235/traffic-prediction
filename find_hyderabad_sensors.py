"""
Identify and extract Hyderabad sensors from the dataset.
Hyderabad is in Telangana state, India - coordinates roughly:
Latitude: 17.36 to 17.45
Longitude: 78.35 to 78.65
"""

import pandas as pd

# Load sensors
df_sensors = pd.read_csv('sensors.csv')

print("=" * 80)
print("FILTERING HYDERABAD SENSORS")
print("=" * 80)

# Hyderabad bounding box
hyderabad_coords = {
    'lat_min': 17.30,
    'lat_max': 17.50,
    'lon_min': 78.30,
    'lon_max': 78.70
}

# Filter Hyderabad sensors
hyderabad_sensors = df_sensors[
    (df_sensors['latitude'] >= hyderabad_coords['lat_min']) &
    (df_sensors['latitude'] <= hyderabad_coords['lat_max']) &
    (df_sensors['longitude'] >= hyderabad_coords['lon_min']) &
    (df_sensors['longitude'] <= hyderabad_coords['lon_max'])
].copy()

print(f"\nTotal sensors in dataset: {len(df_sensors)}")
print(f"Hyderabad sensors found: {len(hyderabad_sensors)}")

if len(hyderabad_sensors) == 0:
    print("\nNo sensors found with strict bounds. Expanding search...")
    hyderabad_sensors = df_sensors[
        (df_sensors['latitude'] >= 17.20) &
        (df_sensors['latitude'] <= 17.60) &
        (df_sensors['longitude'] >= 78.20) &
        (df_sensors['longitude'] <= 78.80)
    ].copy()
    print(f"Hyderabad sensors (expanded): {len(hyderabad_sensors)}")

print("\nHyderabad Sensors Statistics:")
print(f"  Latitude range: {hyderabad_sensors['latitude'].min():.4f} to {hyderabad_sensors['latitude'].max():.4f}")
print(f"  Longitude range: {hyderabad_sensors['longitude'].min():.4f} to {hyderabad_sensors['longitude'].max():.4f}")

print("\nFirst 10 Hyderabad sensors:")
print(hyderabad_sensors[['sensor_id', 'latitude', 'longitude']].head(10))

# Get sensor IDs
hyderabad_sensor_ids = hyderabad_sensors['sensor_id'].tolist()

# Check which are in traffic data
df_traffic = pd.read_csv('traffic_time_series.csv')
available_hyderbad = [s for s in hyderabad_sensor_ids if s in df_traffic['sensor_id'].unique()]

print(f"\nHyderabad sensors with traffic data: {len(available_hyderbad)}")

# Save for training
with open('hyderabad_sensors.txt', 'w') as f:
    for s in hyderabad_sensor_ids:
        f.write(f"{s}\n")

print("\nSaved all Hyderabad sensors to hyderabad_sensors.txt")

# Show distribution
print("\nHyderabad sensors distribution:")
print(hyderabad_sensors[['latitude', 'longitude']].describe())
