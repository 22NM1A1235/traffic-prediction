"""
Direct testing of Flask API predictions for different Hyderabad locations.
This script logs in, makes API calls to /prediction endpoint, and compares results.
"""

import requests
import json
import re
import time

BASE_URL = "http://localhost:5000"
SESSION = requests.Session()

print("=" * 90)
print("TESTING FLASK API PREDICTIONS FOR DIFFERENT LOCATIONS")
print("=" * 90)

# Step 1: Register user
print("\n[1/3] Setting up test account...")
try:
    # Try to register
    resp = SESSION.post(f"{BASE_URL}/register", data={
        'reg_username': 'testuser_hyderbad',
        'reg_password': 'test123456',
        'reg_confirm_password': 'test123456'
    }, timeout=5)
    print("     • Registration attempt completed")
except Exception as e:
    print(f"     ⚠ Registration error (may already exist): {e}")

# Step 2: Login
print("\n[2/3] Authenticating...")
try:
    login_resp = SESSION.post(f"{BASE_URL}/login", data={
        'username': 'testuser_hyderbad',
        'password': 'test123456'
    }, timeout=5)
    
    if login_resp.status_code == 200:
        if 'dashboard' in login_resp.text or 'prediction' in login_resp.text:
            print("     ✓ Login successful")
        else:
            print("     ⚠ Login returned 200 but unexpected content")
    else:
        print(f"     ✗ Login failed with status {login_resp.status_code}")
except Exception as e:
    print(f"     ✗ Login error: {e}")

# Step 3: Test predictions for different Hyderabad locations
print("\n[3/3] Testing API Predictions...")
print("-" * 90)

locations = {
    "Banjara Hills": (17.3850, 78.4867),
    "Gachibowli": (17.3589, 78.5941),
    "Kukatpally": (17.3689, 78.3800),
    "HITEC City": (17.3595, 78.5889),
}

results = {}

print(f"\n{'Location':<20} {'Latitude':<12} {'Longitude':<12} {'Status':<10} {'Response Time':<15}")
print("-" * 90)

for loc_name, (lat, lng) in locations.items():
    try:
        start_time = time.time()
        
        resp = SESSION.post(f"{BASE_URL}/prediction", data={
            'latitude': str(lat),
            'longitude': str(lng)
        }, timeout=10)
        
        elapsed = time.time() - start_time
        
        # Extract plot data or error message from response
        if resp.status_code == 200:
            # Try to find prediction info in the HTML
            if "Traffic Prediction" in resp.text or "plot_url" in resp.text:
                status = "✓ Success"
                # Try to extract the plot image data (base64 encoded)
                plot_match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', resp.text)
                if plot_match:
                    plot_data = plot_match.group(1)
                    # Check plot size as proxy for different predictions
                    results[loc_name] = {
                        'status': 'success',
                        'plot_size': len(plot_data),
                        'lat': lat,
                        'lng': lng
                    }
                else:
                    results[loc_name] = {
                        'status': 'success',
                        'plot_size': 'N/A',
                        'lat': lat,
                        'lng': lng
                    }
            else:
                status = "⚠ Redirect"
                results[loc_name] = {
                    'status': 'redirect',
                    'lat': lat,
                    'lng': lng
                }
        else:
            status = f"✗ {resp.status_code}"
            results[loc_name] = {
                'status': f'error_{resp.status_code}',
                'lat': lat,
                'lng': lng
            }
        
        print(f"{loc_name:<20} {lat:<12.4f} {lng:<12.4f} {status:<10} {elapsed:<15.3f}s")
        
    except Exception as e:
        print(f"{loc_name:<20} {lat:<12.4f} {lng:<12.4f} {'✗ Error':<10} {str(e)[:30]}")
        results[loc_name] = {
            'status': 'error',
            'error': str(e)
        }
    
    time.sleep(0.5)  # Rate limiting

print("\n" + "-" * 90)

# Analyze results
print("\nRESULTS SUMMARY:")
print("-" * 90)

successful_requests = [r for r in results.values() if r['status'] == 'success']
print(f"\nSuccessful requests: {len(successful_requests)}/{len(locations)}")

if len(successful_requests) > 1:
    # Check if plot sizes are different (proxy for different predictions)
    plot_sizes = [r['plot_size'] for r in successful_requests if r['plot_size'] != 'N/A']
    
    if plot_sizes:
        print(f"\nPlot sizes (proxy for different predictions):")
        for loc_name, result in results.items():
            if result['status'] == 'success' and result['plot_size'] != 'N/A':
                print(f"  {loc_name:<20}: {result['plot_size']:<8} bytes")
        
        if len(set(plot_sizes)) > 1:
            print("\n✅ DIFFERENT PLOT SIZES DETECTED - Predictions likely differ!")
        else:
            print("\n⚠ All plots same size - may indicate identical predictions")

print("\nNEXT STEPS:")
print("1. Open http://localhost:5000 in your browser")
print("2. Login with testuser_hyderbad / test123456")
print("3. Navigate to Prediction")
print("4. Test with different Hyderabad coordinates:")
for loc_name, (lat, lng) in locations.items():
    print(f"   • {loc_name}: {lat:.4f}, {lng:.4f}")
print("5. Compare the generated traffic prediction plots visually")

print("\n" + "=" * 90)
