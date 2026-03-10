import requests
from requests.sessions import Session
import time
import re

session = Session()
BASE_URL = "http://127.0.0.1:5000"

# Login  
print("Logging in...")
login_data = {
    'username': 'testuser123',
    'password': 'testpass123'
}
session.post(f"{BASE_URL}/login", data=login_data, timeout=10)

# Test multiple locations
test_locations = [
    {'lat': 17.35, 'lng': 78.50, 'name': 'Location 1'},
    {'lat': 17.36, 'lng': 78.51, 'name': 'Location 2 (nearby)'},
    {'lat': 17.40, 'lng': 78.55, 'name': 'Location 3 (far)'},
    {'lat': 17.50, 'lng': 78.60, 'name': 'Location 4 (very far)'},
]

print("\nTesting Location Differentiation:")
print("=" * 80)

predictions_dict = {}

for loc in test_locations:
    pred_data = {
        'latitude': str(loc['lat']),
        'longitude': str(loc['lng'])
    }
    
    resp = session.post(f"{BASE_URL}/prediction", data=pred_data, timeout=30)
    
    if resp.status_code == 200 and 'PREDICTION STATISTICS' in resp.text:
        # Extract prediction values from response
        # Look for the mean flow value
        match = re.search(r'AVERAGE FLOW.*?<div style.*?>(\d+\.\d+)</div>', resp.text, re.DOTALL)
        if match:
            mean_val = float(match.group(1))
            predictions_dict[loc['name']] = mean_val
            print(f"\n{loc['name']} ({loc['lat']}, {loc['lng']})")
            print(f"  Average Flow: {mean_val} vehicles/hour")
        else:
            print(f"\n{loc['name']} - Could not extract prediction value")
    else:
        print(f"\n{loc['name']} - Request failed or no prediction data")

print("\n" + "=" * 80)
print("Summary:")
if len(predictions_dict) >= 2:
    all_predictions = list(predictions_dict.values())
    if len(set(all_predictions)) > 1:
        print("✓ LOCATION DIFFERENTIATION WORKING: Different locations have different predictions")
        for name, val in predictions_dict.items():
            print(f"  {name}: {val}")
    else:
        print("✗ LOCATION DIFFERENTIATION NOT WORKING: All predictions are the same")
        for name, val in predictions_dict.items():
            print(f"  {name}: {val}")
else:
    print(f"Could only get {len(predictions_dict)} prediction(s), need at least 2 to compare")
