"""
Quick test - show all predictions at once
"""
import requests
import re

session = requests.Session()
session.post("http://localhost:5000/login", data={"username": "test"})

locations = [
    ("Cyberabad", 17.4400, 78.6100),
    ("Kukatpally", 17.3821, 78.4184),
]

print("\n" + "="*80)
print("LOCATION-SPECIFIC TRAFFIC PREDICTIONS")
print("="*80 + "\n")

results = []

for loc_name, lat, lon in locations:
    response = session.post("http://localhost:5000/prediction",
                           data={"latitude": str(lat), "longitude": str(lon)},
                           allow_redirects=True,
                           timeout=10)
    
    # Extract the numbers
    avg_match = re.search(r'AVERAGE FLOW.*?<div[^>]*>(\d+\.\d+)</div>', response.text, re.DOTALL)
    if avg_match:
        avg = float(avg_match.group(1))
        results.append((loc_name, lat, lon, avg))
        print(f"{loc_name:20} ({lat:.4f}, {lon:.4f})  →  {avg:.2f} vehicles/hour")

print("\n" + "="*80)
if len(results) >= 2:
    values = [r[3] for r in results]
    diff = abs(values[0] - values[1])
    pct = (diff / max(values)) * 100
    print(f"\nDifference: {diff:.2f} vehicles/hour ({pct:.1f}%)")
    
    if diff > 0.5:
        print("RESULT: SUCCESS - Different locations produce DIFFERENT predictions!")
    else:
        print("RESULT: Predictions are very similar")
else:
    print("Could not extract enough predictions")

print("="*80 + "\n")
