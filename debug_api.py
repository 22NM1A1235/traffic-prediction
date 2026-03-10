"""
Debug test to see why predictions are getting redirected.
"""

import requests
import time

BASE_URL = "http://localhost:5000"
SESSION = requests.Session()

print("=" * 80)
print("DEBUG: Testing API with Error Capture")
print("=" * 80)

# Register and login
print("\n[Step 1] Register...")
reg_resp = SESSION.post(f"{BASE_URL}/register", data={
    'reg_username': 'debug_user',
    'reg_password': 'pass123',
    'reg_confirm_password': 'pass123'
})
print(f"  Status: {reg_resp.status_code}")

print("\n[Step 2] Login...")
login_resp = SESSION.post(f"{BASE_URL}/login", data={
    'username': 'debug_user', 
    'password': 'pass123'
})
print(f"  Status: {login_resp.status_code}")
print(f"  Has 'dashboard': {'dashboard' in login_resp.text}")
print(f"  Has 'prediction': {'prediction' in login_resp.text}")

# Check cookies
print(f"  Cookies: {SESSION.cookies.get_dict()}")

print("\n[Step 3] Test GET /prediction (initial page)...")
get_resp = SESSION.get(f"{BASE_URL}/prediction")
print(f"  Status: {get_resp.status_code}")

time.sleep(1)

print("\n[Step 4] Make prediction request...")
pred_data = {
    'latitude': '17.3850',
    'longitude': '78.4867'
}

pred_resp = SESSION.post(f"{BASE_URL}/prediction", data=pred_data, allow_redirects=False)
print(f"  Status: {pred_resp.status_code}")
print(f"  Redirect location: {pred_resp.headers.get('Location', 'N/A')}")

if pred_resp.status_code == 302 or pred_resp.status_code == 301:
    print("\n  Got redirect! Following redirect...")
    redirect_url = pred_resp.headers.get('Location')
    if redirect_url:
        if not redirect_url.startswith('http'):
            redirect_url = BASE_URL + redirect_url
        print(f"  Redirect URL: {redirect_url}")
        
        final_resp = SESSION.get(redirect_url)
        print(f"  Final status: {final_resp.status_code}")
        
        # Look for flash messages (error messages)
        if "Alert" in final_resp.text or "alert" in final_resp.text:
            # Extract the text around alert
            lines = final_resp.text.split('\n')
            for i, line in enumerate(lines):
                if 'alert' in line.lower() or 'error' in line.lower():
                    print(f"  Line {i}: {line[:200]}")
else:
    print("\n  Got direct response (not redirect)")
    if "Traffic Prediction" in pred_resp.text:
        print("  ✓ Got prediction page!")
    else:
        print("  ? Unexpected response")

print("\n" + "=" * 80)
