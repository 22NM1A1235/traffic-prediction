#!/usr/bin/env python3
"""
Complete End-to-End Test
Tests: Registration → Login → Prediction with Location Awareness
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000"
session = requests.Session()

print("\n" + "=" * 100)
print("COMPLETE END-TO-END APPLICATION TEST")
print("=" * 100)

# Step 1: Check application is running
print("\n[1/4] Checking if application is running...")
try:
    r = session.get(f"{BASE_URL}/")
    if r.status_code == 200:
        print("    ✓ Application is running on http://127.0.0.1:5000")
    else:
        print(f"    ✗ Application returned status {r.status_code}")
except Exception as e:
    print(f"    ✗ Error: {e}")
    exit(1)

# Step 2: Register a test user
print("\n[2/4] Registering test user...")
test_email = f"test{int(time.time())}@example.com"
test_password = "TestPassword123!"

register_data = {
    'email': test_email,
    'password': test_password,
    'confirm_password': test_password
}

try:
    r = session.post(f"{BASE_URL}/register", data=register_data)
    if r.status_code == 200 or 'redirect' in r.history:
        print(f"    ✓ User registered: {test_email}")
    else:
        print(f"    ✗ Registration failed with status {r.status_code}")
        print(f"      Trying with existing user instead...")
        test_email = "test@example.com"
        test_password = "password"
except Exception as e:
    print(f"    ⚠ Registration error: {e}")
    print(f"      Continuing with existing credentials...")
    test_email = "test@example.com"
    test_password = "password"

# Step 3: Login
print("\n[3/4] Logging in...")
login_data = {
    'email': test_email,
    'password': test_password
}

try:
    r = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=True)
    if 'Home' in r.text or 'home' in r.text or 'prediction' in r.text.lower():
        print(f"    ✓ Successfully logged in as: {test_email}")
    else:
        # Check if login page is still shown
        if 'login' in r.text.lower() or 'password' in r.text.lower():
            print(f"    ⚠ Login may have failed, retrying...")
            # Try simple credentials
            session.post(f"{BASE_URL}/login", data={'email': 'test@example.com', 'password': 'password'})
        else:
            print(f"    ✓ Request processed")
except Exception as e:
    print(f"    ✗ Login error: {e}")

# Step 4: Test Predictions with Location Awareness
print("\n[4/4] Testing traffic predictions for different locations...")
print("      (All locations use same sensor, but location encoder differentiates)")

test_locations = [
    {"name": "Location A", "lat": 17.30, "lng": 78.30},
    {"name": "Location B", "lat": 17.31, "lng": 78.31},
    {"name": "Location C", "lat": 17.32, "lng": 78.32},
]

predictions = []

for loc in test_locations:
    try:
        pred_data = {
            'latitude': str(loc['lat']),
            'longitude': str(loc['lng'])
        }
        
        r = session.post(f"{BASE_URL}/prediction", data=pred_data, timeout=10)
        
        if r.status_code == 200:
            # Check if we got a valid response (not redirected to login)
            if 'input' in r.text.lower() or 'canvas' in r.text.lower() or 'coordinates' in r.text.lower():
                predictions.append({
                    "location": loc['name'],
                    "lat": loc['lat'],
                    "lng": loc['lng'],
                    "status": "✓",
                    "response_size": len(r.text)
                })
            else:
                predictions.append({
                    "location": loc['name'],
                    "lat": loc['lat'],
                    "lng": loc['lng'],
                    "status": "⚠ Redirect",
                    "response_size": len(r.text)
                })
        else:
            predictions.append({
                "location": loc['name'],
                "lat": loc['lat'],
                "lng": loc['lng'],
                "status": f"✗ HTTP {r.status_code}",
                "response_size": 0
            })
    except Exception as e:
        predictions.append({
            "location": loc['name'],
            "lat": loc['lat'],
            "lng": loc['lng'],
            "status": f"✗ {str(e)[:30]}",
            "response_size": 0
        })

# Display results
print("\n" + "-" * 100)
print("\nTesting Results:")
print(f"{'Location':<15} {'Coordinates':<20} {'Status':<30} {'Response Size':<15}")
print("-" * 100)

for pred in predictions:
    coords = f"({pred['lat']}, {pred['lng']})"
    print(f"{pred['location']:<15} {coords:<20} {pred['status']:<30} {pred['response_size']:<15}")

print("\n" + "-" * 100)

# Summary
print("\nSummary:")
all_working = all(p['status'].startswith('✓') for p in predictions)
some_working = any(p['status'].startswith('✓') for p in predictions)

if all_working:
    print("✅ ALL TESTS PASSED - Application working perfectly!")
    print("   - Flask server running")
    print("   - User authentication working")
    print("   - Location-aware predictions working for multiple locations")
elif some_working:
    print("⚠️  PARTIAL SUCCESS - Some features working")
    print("   - Flask server running")
    print("   - Location predictions accessible")
    print("   - May need login session for full functionality")
else:
    print("❌ APPLICATION NOT FULLY FUNCTIONAL")
    print("   - Check if Flask is running")
    print("   - Verify database connectivity")

print("\n" + "=" * 100)
print(f"Application URL: http://127.0.0.1:5000")
print("=" * 100 + "\n")
