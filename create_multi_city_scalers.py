import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load sensor data
data = pd.read_csv('sensors.csv')

# Define city boundaries
bangalore_lat = 12.9352  # center
bangalore_lon = 77.6245  # center
hyderabad_lat = 17.3850  # center
hyderabad_lon = 78.4867  # center

# Classify sensors into cities based on proximity
def classify_city(lat, lon):
    dist_bangalore = np.sqrt((lat - bangalore_lat)**2 + (lon - bangalore_lon)**2)
    dist_hyderabad = np.sqrt((lat - hyderabad_lat)**2 + (lon - hyderabad_lon)**2)
    
    if dist_bangalore < dist_hyderabad:
        return 'Bangalore'
    else:
        return 'Hyderabad'

data['city'] = data.apply(lambda row: classify_city(row['latitude'], row['longitude']), axis=1)

print('City Classification:')
print(data['city'].value_counts())
print()

# Create city-specific scalers
cities = ['Bangalore', 'Hyderabad']
city_scalers = {}
city_centers = {}

for city in cities:
    city_data = data[data['city'] == city]
    coords = city_data[['latitude', 'longitude']].values
    
    # Create scaler for this city
    scaler = StandardScaler()
    scaler.fit(coords)
    city_scalers[city] = scaler
    
    # Store center
    center_lat = city_data['latitude'].mean()
    center_lon = city_data['longitude'].mean()
    city_centers[city] = (center_lat, center_lon)
    
    print(f'{city}:')
    print(f'  Sensors: {len(city_data)}')
    print(f'  Center: ({center_lat:.4f}, {center_lon:.4f})')
    print(f'  Scaler mean: {scaler.mean_}')
    print(f'  Scaler scale: {scaler.scale_}')
    print()

# Save scalers and city info
with open('saved_models/city_scalers.pkl', 'wb') as f:
    pickle.dump(city_scalers, f)

with open('saved_models/city_centers.pkl', 'wb') as f:
    pickle.dump(city_centers, f)

print('✓ City-specific scalers created and saved!')
print('  - saved_models/city_scalers.pkl')
print('  - saved_models/city_centers.pkl')
