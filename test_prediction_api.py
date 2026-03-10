import requests
from requests.sessions import Session
import time

# Wait for server to be ready
time.sleep(2)

# Create session to maintain cookies
session = Session()

BASE_URL = "http://127.0.0.1:5000"

# Step 1: Register a test user (if not exists)
print("Step 1: Attempting to register test user...")
register_data = {
    'username': 'testuser123',
    'password': 'testpass123',
    'confirm_password': 'testpass123'
}
try:
    resp = session.post(f"{BASE_URL}/register", data=register_data, timeout=10)
    print(f"  Register response: {resp.status_code}")
except Exception as e:
    print(f"  Register failed (user may already exist): {e}")

# Step 2: Login
print("\nStep 2: Logging in...")
login_data = {
    'username': 'testuser123',
    'password': 'testpass123'
}
resp = session.post(f"{BASE_URL}/login", data=login_data, timeout=10)
print(f"  Login response: {resp.status_code}")

# Step 3: Make a prediction
print("\nStep 3: Making prediction request...")
pred_data = {
    'latitude': '17.35',
    'longitude': '78.50'
}
resp = session.post(f"{BASE_URL}/prediction", data=pred_data, timeout=30)
print(f"  Prediction response: {resp.status_code}")
print(f"  Content length: {len(resp.text)}")

if resp.status_code == 200:
    # Check if it has prediction content or redirect
    if 'Login - TrafficAI' in resp.text:
        print("  ERROR: Redirected to login (session not maintained)")
    elif 'PREDICTION STATISTICS' in resp.text or 'Queried Location' in resp.text:
        print("  SUCCESS: Prediction page rendered")
        print(f"  First 500 chars: {resp.text[:500]}")
    elif 'Error' in resp.text or 'error' in resp.text:
        print("  ERROR: Prediction returned an error")
        # Find error messages
        if '<div class="alert' in resp.text:
            idx = resp.text.find('<div class="alert')
            print(f"  Error content: {resp.text[idx:idx+500]}")
    else:
        print("  Response appears to be something else")
        print(f"  First 1000 chars: {resp.text[:1000]}")
else:
    print(f"  ERROR: Got status {resp.status_code}")
    print(f"  Response: {resp.text[:500]}")

# Check debug log
print("\n\nDEBUG LOG:")
try:
    with open('prediction_debug.log', 'r') as f:
        log_content = f.read()
        # Print last 2000 characters
        print(log_content[-2000:])
except Exception as e:
    print(f"Could not read debug log: {e}")
