"""
Simple location test - extract actual predictions from HTML
"""
import requests
import re

BASE_URL = "http://localhost:5000"

print("\nTesting different Hyderabad locations...\n")

# Create a persistent session
session = requests.Session()

# Login
print("Logging in...")
session.post(f"{BASE_URL}/login", data={"username": "test"})

# Test locations
locations = [
    ("Cyberabad", 17.4400, 78.6100),
    ("Kukatpally", 17.3821, 78.4184),
    ("HITEC City", 17.3532, 78.4990),
]

for name, lat, lon in locations:
    print(f"\nTesting {name} ({lat:.4f}, {lon:.4f})...")
    
    try:
        response = session.post(
            f"{BASE_URL}/prediction",
            data={"latitude": str(lat), "longitude": str(lon)},
            allow_redirects=True,
            timeout=10
        )
        
        # Check if plot is in response
        if "plot_url" in response.text:
            print(f"  - PLOT GENERATED: YES")
            
            # Extract some info
            if "<title>" in response.text:
                title = response.text.split("<title>")[1].split("</title>")[0]
                print(f"  - Page Title: {title}")
        else:
            print(f"  - PLOT GENERATED: NO")
            print(f"  - Response length: {len(response.text)}")
            
    except Exception as e:
        print(f"  - ERROR: {e}")

print("\nDone!")
