import requests
import time

BASE_URL = "http://localhost:5000"

# First, login
print("Logging in...")
session = requests.Session()
login_data = {
    'username': 'testuser',
    'password': 'testpass'
}
# Try to create account first
account_data = {
    'reg_username': 'testuser',
    'reg_password': 'testpass',
    'reg_confirm_password': 'testpass'
}
try:
    # Register
    resp = session.post(f"{BASE_URL}/register", data=account_data)
    print(f"Register status: {resp.status_code}")
except:
    pass

# Now login
resp = session.post(f"{BASE_URL}/login", data=login_data)
print(f" Login status: {resp.status_code}")

# Test different Hyderabad locations
locations = {
    "Location A (Banjara Hills)": (17.3850, 78.4867),
    "Location B (Gachibowli)": (17.3589, 78.5941),
    "Location C (Kukatpally)": (17.3689, 78.3800),
    "Location D (HITEC City)": (17.3595, 78.5889),
}

predictions = {}

print("\n" + "=" * 80)
print("TESTING FLASK PREDICTIONS FOR DIFFERENT LOCATIONS")
print("=" * 80)

for location_name, (lat, lng) in locations.items():
    print(f"\nTesting {location_name}...")
    
    pred_data = {
        'latitude': str(lat),
        'longitude': str(lng)
    }
    
    resp = session.post(f"{BASE_URL}/prediction", data=pred_data)
    print(f"  Response status: {resp.status_code}")
    
    # Try to extract prediction values from the page
    if "Traffic Prediction" in resp.text:
        # Extract the plot data if available
        predictions[location_name] = resp.status_code
        print(f"  ✓ Got prediction page")
    else:
        print(f"  ✗ No prediction found (likely redirect or error)")
    
    time.sleep(0.5)

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print("Check http://localhost:5000/ to see if different locations have different plots")
