"""
Create enhanced location features for better location differentiation
Includes: relative position features, district info, and descriptive metadata
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

print("Creating enhanced location features for Hyderabad...")

# Load sensors
df_sensors = pd.read_csv('sensors.csv')

# Hyderabad center
HYDERABAD_CENTER_LAT = 17.3850
HYDERABAD_CENTER_LON = 78.4867

# Define city quadrants and zones
def assign_zone(lat, lon):
    """Assign zone based on position relative to city center"""
    if lat > HYDERABAD_CENTER_LAT and lon > HYDERABAD_CENTER_LON:
        return "NE"  # Northeast
    elif lat > HYDERABAD_CENTER_LAT and lon <= HYDERABAD_CENTER_LON:
        return "NW"  # Northwest
    elif lat <= HYDERABAD_CENTER_LAT and lon > HYDERABAD_CENTER_LON:
        return "SE"  # Southeast
    else:
        return "SW"  # Southwest

def assign_district(lat, lon):
    """Rough district assignment based on coordinates"""
    # This is approximate based on known Hyderabad districts
    if lat > 17.4:
        if lon > 78.6:
            return "MEDCHAL"
        else:
            return "SECUNDERABAD"
    elif lat > 17.35:
        if lon > 78.5:
            return "WHITEFIELD"
        else:
            return "HYDERABAD"
    elif lat > 17.3:
        if lon > 78.5:
            return "LB_NAGAR"
        else:
            return "HYDERABAD"
    else:
        return "OUTER"

# Create enhanced location features
df_sensors['zone'] = df_sensors.apply(lambda r: assign_zone(r['latitude'], r['longitude']), axis=1)
df_sensors['district'] = df_sensors.apply(lambda r: assign_district(r['latitude'], r['longitude']), axis=1)

# Compute relative position features
df_sensors['lat_offset'] = df_sensors['latitude'] - HYDERABAD_CENTER_LAT
df_sensors['lon_offset'] = df_sensors['longitude'] - HYDERABAD_CENTER_LON
df_sensors['distance_from_center'] = np.sqrt(
    df_sensors['lat_offset']**2 + df_sensors['lon_offset']**2
)

# Normalize the offsets and distance for model input
lat_offset_min = df_sensors['lat_offset'].min()
lat_offset_max = df_sensors['lat_offset'].max()
lon_offset_min = df_sensors['lon_offset'].min()
lon_offset_max = df_sensors['lon_offset'].max()
dist_min = df_sensors['distance_from_center'].min()
dist_max = df_sensors['distance_from_center'].max()

df_sensors['lat_offset_norm'] = (df_sensors['lat_offset'] - lat_offset_min) / (lat_offset_max - lat_offset_min)
df_sensors['lon_offset_norm'] = (df_sensors['lon_offset'] - lon_offset_min) / (lon_offset_max - lon_offset_min)
df_sensors['distance_norm'] = (df_sensors['distance_from_center'] - dist_min) / (dist_max - dist_min)

# Create quadrant encoding (one-hot: N/S/E/W)
df_sensors['is_north'] = (df_sensors['latitude'] > HYDERABAD_CENTER_LAT).astype(float)
df_sensors['is_east'] = (df_sensors['longitude'] > HYDERABAD_CENTER_LON).astype(float)

print("\n" + "="*60)
print("ENHANCED LOCATION FEATURES")
print("="*60)

print(f"\nZone distribution:")
print(df_sensors['zone'].value_counts())

print(f"\nDistrict distribution:")
print(df_sensors['district'].value_counts())

print(f"\nRelative position ranges:")
print(f"  Latitude offset: {lat_offset_min:.4f} to {lat_offset_max:.4f}")
print(f"  Longitude offset: {lon_offset_min:.4f} to {lon_offset_max:.4f}")
print(f"  Distance from center: {dist_min:.4f} to {dist_max:.4f}")

# Save location lookup dictionary for app.py to use
location_info = {}
for idx, row in df_sensors.iterrows():
    sensor_id = row['sensor_id']
    location_info[sensor_id] = {
        'zone': row['zone'],
        'district': row['district'],
        'lat_offset': float(row['lat_offset']),
        'lon_offset': float(row['lon_offset']),
        'distance_from_center': float(row['distance_from_center']),
        'is_north': float(row['is_north']),
        'is_east': float(row['is_east'])
    }

with open('saved_models/location_info.pkl', 'wb') as f:
    pickle.dump(location_info, f)

print(f"\n✓ Saved location info for {len(location_info)} sensors")
print(f"✓ Location features file: saved_models/location_info.pkl")

# Also save normalization parameters for real-time queries
norm_params = {
    'center_lat': HYDERABAD_CENTER_LAT,
    'center_lon': HYDERABAD_CENTER_LON,
    'lat_offset_min': float(lat_offset_min),
    'lat_offset_max': float(lat_offset_max),
    'lon_offset_min': float(lon_offset_min),
    'lon_offset_max': float(lon_offset_max),
    'dist_min': float(dist_min),
    'dist_max': float(dist_max),
}

with open('saved_models/location_norm_params.pkl', 'wb') as f:
    pickle.dump(norm_params, f)

print(f"✓ Saved normalization parameters: saved_models/location_norm_params.pkl")

# Test with different locations
print("\n" + "="*60)
print("TESTING LOCATION FEATURE EXTRACTION")
print("="*60)

test_locations = [
    ("Downtown", 17.3850, 78.4867),
    ("Uptown", 17.4500, 78.5200),
    ("Far East", 17.5000, 78.6000),
    ("South", 17.3000, 78.4500),
]

print("\nLocation feature comparison:")
print("-" * 80)
for name, lat, lon in test_locations:
    lat_off = lat - HYDERABAD_CENTER_LAT
    lon_off = lon - HYDERABAD_CENTER_LON
    dist = np.sqrt(lat_off**2 + lon_off**2)
    lat_off_norm = (lat_off - lat_offset_min) / (lat_offset_max - lat_offset_min)
    lon_off_norm = (lon_off - lon_offset_min) / (lon_offset_max - lon_offset_min)
    dist_norm = (dist - dist_min) / (dist_max - dist_min)
    zone = assign_zone(lat, lon)
    district = assign_district(lat, lon)
    
    print(f"\n{name:15s} ({lat:.4f}, {lon:.4f})")
    print(f"  Zone: {zone}, District: {district}")
    print(f"  Offsets: lat={lat_off:+.4f}, lon={lon_off:+.4f}")
    print(f"  Normalized: lat={lat_off_norm:.4f}, lon={lon_off_norm:.4f}, dist={dist_norm:.4f}")
    print(f"  Feature vector: [{lat_off_norm:.4f}, {lon_off_norm:.4f}, {dist_norm:.4f}]")

print("\n" + "="*60)
print("FEATURE EXTRACTION COMPLETE")
print("="*60)
