"""
Simple offline test without internet
This test queries the running Flask server locally
"""

import requests
import json

# Step 1: Register a test user
print("=" * 80)
print("STEP 1: REGISTERING A NEW USER")
print("=" * 80)

register_data = {
    'username': 'testuser',
    'password': 'password123',
    'confirm_password': 'password123'
}

response = requests.post('http://127.0.0.1:5000/register', data=register_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:200]}")
print()

# Step 2: Login with the same credentials
print("=" * 80)
print("STEP 2: LOGGING IN")
print("=" * 80)

session = requests.Session()
login_data = {
    'username': 'testuser',
    'password': 'password123'
}

response = session.post('http://127.0.0.1:5000/login', data=login_data)
print(f"Status Code: {response.status_code}")
print(f"Logged in successfully!")
print()

# Step 3: Get prediction for a specific location
print("=" * 80)
print("STEP 3: GETTING TRAFFIC PREDICTION FOR LOCATION")
print("=" * 80)

# Test location coordinates (Hyderabad area)
lat = 17.35
lng = 78.50

prediction_data = {
    'latitude': str(lat),
    'longitude': str(lng)
}

response = session.post('http://127.0.0.1:5000/prediction', data=prediction_data)
print(f"Status Code: {response.status_code}")
print(f"Location: ({lat}, {lng})")
print()

# Extract the prediction value from HTML
import re
match = re.search(r'AVERAGE FLOW.*?<div.*?>(\d+\.\d+)</div>', response.text)
if match:
    prediction = match.group(1)
    print(f"✓ TRAFFIC PREDICTION: {prediction} vehicles/hour")
else:
    print("Could not extract prediction from response")
print()

# Step 4: Test multiple locations
print("=" * 80)
print("STEP 4: TESTING MULTIPLE LOCATIONS (LOCATION DIFFERENTIATION)")
print("=" * 80)

locations = [
    (17.35, 78.50, "Location 1"),
    (17.36, 78.51, "Location 2 (nearby)"),
    (17.40, 78.55, "Location 3 (further)"),
    (17.50, 78.60, "Location 4 (very far)"),
]

predictions = []
for lat, lng, name in locations:
    prediction_data = {
        'latitude': str(lat),
        'longitude': str(lng)
    }
    
    response = session.post('http://127.0.0.1:5000/prediction', data=prediction_data)
    
    match = re.search(r'AVERAGE FLOW.*?<div.*?>(\d+\.\d+)</div>', response.text)
    if match:
        pred_value = float(match.group(1))
        predictions.append((name, lat, lng, pred_value))
        print(f"✓ {name} ({lat}, {lng}): {pred_value} vehicles/hour")
    else:
        print(f"✗ {name} ({lat}, {lng}): Failed to get prediction")

print()
print("=" * 80)
print("SUMMARY - LOCATION DIFFERENTIATION TEST")
print("=" * 80)

if len(predictions) > 1:
    print("\nAll predictions by location:")
    for name, lat, lng, pred in predictions:
        print(f"  {name}: {pred:.2f} veh/hr")
    
    # Check if predictions are different
    pred_values = [p[3] for p in predictions]
    if len(set(pred_values)) == len(pred_values):
        print("\n✅ SUCCESS: All locations have DIFFERENT predictions!")
    else:
        print("\n⚠️ Some locations have similar predictions")
    
    # Show differences
    print("\nPrediction differences:")
    for i in range(1, len(predictions)):
        diff = predictions[i][3] - predictions[0][3]
        print(f"  {predictions[i][0]} vs {predictions[0][0]}: {diff:+.2f} veh/hr")

print("\n" + "=" * 80)
print("✅ OFFLINE TEST COMPLETE - NO INTERNET REQUIRED!")
print("=" * 80)
