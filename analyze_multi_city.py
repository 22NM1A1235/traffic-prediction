import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load sensor data
data = pd.read_csv('sensors.csv')
coords = data[['latitude', 'longitude']].values

# Use K-means to find city centers
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(coords)

# Analyze each cluster
print('DETECTED CITY CLUSTERS:')
print('='*70)
for cluster_id in range(5):
    cluster_data = data[data['cluster'] == cluster_id]
    lat_mean = cluster_data['latitude'].mean()
    lon_mean = cluster_data['longitude'].mean()
    lat_min, lat_max = cluster_data['latitude'].min(), cluster_data['latitude'].max()
    lon_min, lon_max = cluster_data['longitude'].min(), cluster_data['longitude'].max()
    
    print(f'\nCluster {cluster_id}: {len(cluster_data)} sensors')
    print(f'  Center: ({lat_mean:.4f}, {lon_mean:.4f})')
    print(f'  Lat range: {lat_min:.4f} to {lat_max:.4f}')
    print(f'  Lon range: {lon_min:.4f} to {lon_max:.4f}')
    example_sensors = cluster_data['sensor_id'].iloc[:3].tolist()
    print(f'  Example sensors: {example_sensors}')

# Get specific sensors mentioned
print('\n' + '='*70)
print('SPECIFIC SENSORS MENTIONED:')
s11_s100 = data[(data['sensor_id'].str.extract('(\d+)', expand=False).astype(int) >= 11) & 
                (data['sensor_id'].str.extract('(\d+)', expand=False).astype(int) <= 100)]
s2200_s2300 = data[(data['sensor_id'].str.extract('(\d+)', expand=False).astype(int) >= 2200) & 
                   (data['sensor_id'].str.extract('(\d+)', expand=False).astype(int) <= 2300)]

print(f'\nSensors S11-S100: {len(s11_s100)} sensors')
if len(s11_s100) > 0:
    print(f'  Lat: {s11_s100["latitude"].min():.4f} to {s11_s100["latitude"].max():.4f}')
    print(f'  Lon: {s11_s100["longitude"].min():.4f} to {s11_s100["longitude"].max():.4f}')

print(f'\nSensors S2200-S2300: {len(s2200_s2300)} sensors')
if len(s2200_s2300) > 0:
    print(f'  Lat: {s2200_s2300["latitude"].min():.4f} to {s2200_s2300["latitude"].max():.4f}')
    print(f'  Lon: {s2200_s2300["longitude"].min():.4f} to {s2200_s2300["longitude"].max():.4f}')

# They're in the same cluster, calculate distance
if len(s11_s100) > 0 and len(s2200_s2300) > 0:
    dist = np.sqrt((s11_s100['latitude'].mean() - s2200_s2300['latitude'].mean())**2 + 
                   (s11_s100['longitude'].mean() - s2200_s2300['longitude'].mean())**2)
    print(f'\nDistance between clusters: {dist:.6f} degrees (~{dist*111:.2f}km)')
