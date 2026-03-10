"""
Check actual response content from prediction endpoint.
"""

import requests
import time

BASE_URL = "http://localhost:5000"
SESSION = requests.Session()

# Login (user already exists from previous test)
print("Logging in...")
login_resp = SESSION.post(f"{BASE_URL}/login", data={
    'username': 'debug_user', 
    'password': 'pass123'
})

time.sleep(0.5)

# Make prediction request
print("Making prediction request...")
pred_resp = SESSION.post(f"{BASE_URL}/prediction", data={
    'latitude': '17.3850',
    'longitude': '78.4867'
})

print(f"\nResponse status: {pred_resp.status_code}")
print(f"Response length: {len(pred_resp.text)} characters")
print(f"Response headers: {dict(pred_resp.headers)}")

# Save to file for inspection
with open('prediction_response.html', 'w', encoding='utf-8') as f:
    f.write(pred_resp.text)

print("\nResponse saved to prediction_response.html")

# Print first 2000 chars
print("\nFirst 2000 characters of response:")
print("=" * 80)
print(pred_resp.text[:2000])
print("=" * 80)
