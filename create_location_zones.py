import pickle
import numpy as np
import pandas as pd

# Load sensor data
data = pd.read_csv('sensors.csv')

# Define location zones for both cities
# Hyderabad zones (17.3586°N, 78.4806°E center)
hyderabad_zones = [
    {
        'name': 'HiTech City (Madhapur)',
        'lat': 17.3589,
        'lon': 78.3877,
        'traffic_factor': 1.8,  # Very high traffic - premium IT district
        'description': 'Major IT hub, high-rise office complex'
    },
    {
        'name': 'Gachibowli',
        'lat': 17.3789,
        'lon': 78.4089,
        'traffic_factor': 1.5,  # High traffic
        'description': 'IT and residential mixed area'
    },
    {
        'name': 'Jubilee Hills',
        'lat': 17.3967,
        'lon': 78.4356,
        'traffic_factor': 1.3,  # Medium-high traffic
        'description': 'Upscale residential, commercial'
    },
    {
        'name': 'Begumpet',
        'lat': 17.3720,
        'lon': 78.4833,
        'traffic_factor': 1.1,  # Medium traffic
        'description': 'Central business area'
    },
    {
        'name': 'Kompally',
        'lat': 17.5000,
        'lon': 78.7000,
        'traffic_factor': 0.6,  # Lower traffic - suburban
        'description': 'Suburban, industrial area'
    },
    {
        'name': 'Miyapur',
        'lat': 17.4500,
        'lon': 78.5500,
        'traffic_factor': 0.7,  # Lower traffic
        'description': 'Peripheral residential area'
    },
    {
        'name': 'Secunderabad',
        'lat': 17.3686,
        'lon': 78.5056,
        'traffic_factor': 1.2,  # Medium traffic
        'description': 'Historic commercial area'
    },
]

# Bangalore zones (12.9506°N, 77.5498°E center)
bangalore_zones = [
    {
        'name': 'Whitefield',
        'lat': 12.9698,
        'lon': 77.6994,
        'traffic_factor': 1.7,  # Very high traffic - IT corridor
        'description': 'Major IT hub, heavy congestion'
    },
    {
        'name': 'Electronic City',
        'lat': 12.8441,
        'lon': 77.6748,
        'traffic_factor': 1.6,  # High traffic - IT park
        'description': 'Large IT park, significant traffic'
    },
    {
        'name': 'Indiranagar',
        'lat': 13.0306,
        'lon': 77.6408,
        'traffic_factor': 1.4,  # High traffic
        'description': 'Central, busy commercial area'
    },
    {
        'name': 'Koramangala',
        'lat': 12.9352,
        'lon': 77.6245,
        'traffic_factor': 1.3,  # Medium-high traffic
        'description': 'Trendy commercial and residential'
    },
    {
        'name': 'HSR Layout',
        'lat': 13.0169,
        'lon': 77.6200,
        'traffic_factor': 1.0,  # Medium traffic
        'description': 'Residential area'
    },
    {
        'name': 'Hebbal',
        'lat': 13.0024,
        'lon': 77.5989,
        'traffic_factor': 0.9,  # Moderate traffic
        'description': 'Mixed residential and commercial'
    },
    {
        'name': 'Banaswadi',
        'lat': 13.0428,
        'lon': 77.5898,
        'traffic_factor': 0.8,  # Lower traffic
        'description': 'Residential area'
    },
]

# Write function to find nearest zone for a given location
def classify_location_to_zone(lat, lon, city):
    """Find nearest location zone and return name with traffic characteristics"""
    
    if city == 'Hyderabad':
        zones = hyderabad_zones
    else:
        zones = bangalore_zones
    
    min_dist = float('inf')
    nearest_zone = None
    
    for zone in zones:
        dist = np.sqrt((lat - zone['lat'])**2 + (lon - zone['lon'])**2)
        if dist < min_dist:
            min_dist = dist
            nearest_zone = zone
    
    return nearest_zone

# Save zones data for use in app
zones_data = {
    'Hyderabad': hyderabad_zones,
    'Bangalore': bangalore_zones
}

with open('saved_models/location_zones.pkl', 'wb') as f:
    pickle.dump(zones_data, f)

print('✓ Location zones created:')
print('\nHyderabad zones:')
for zone in hyderabad_zones:
    print(f"  {zone['name']:<30} Traffic Factor: {zone['traffic_factor']:.1f}x")

print('\nBangalore zones:')
for zone in bangalore_zones:
    print(f"  {zone['name']:<30} Traffic Factor: {zone['traffic_factor']:.1f}x")

print('\n✓ Saved to saved_models/location_zones.pkl')
