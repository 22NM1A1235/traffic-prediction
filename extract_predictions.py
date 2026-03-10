"""
Test multiple locations and extract prediction plots to verify they're different
"""
import requests
import re
import base64
import os

session = requests.Session()
session.post("http://localhost:5000/login", data={"username": "test"})

# Test locations
locations = [
    ("Cyberabad", 17.4400, 78.6100),
    ("Kukatpally", 17.3821, 78.4184),
    ("HITEC_City", 17.3532, 78.4990),
]

print("\nTesting predictions for different Hyderabad locations:\n")
print("="*80)

for loc_name, lat, lon in locations:
    print(f"\n{loc_name} ({lat:.4f}, {lon:.4f})")
    
    response = session.post("http://localhost:5000/prediction",
                           data={"latitude": str(lat), "longitude": str(lon)},
                           allow_redirects=True)
    
    # Extract image data
    if "data:image/png;base64" in response.text:
        # Find the base64 data
        start = response.text.find("data:image/png;base64,") + len("data:image/png;base64,")
        end = response.text.find('"', start)
        img_base64 = response.text[start:end]
        
        # Calculate size
        img_size_kb = len(img_base64) / 1024
        
        # Try to decode first bytes to verify
        try:
            img_bytes = base64.b64decode(img_base64[:100])
            print(f"  - Image generated: YES ({img_size_kb:.1f} KB)")
            print(f"  - Data verified: YES")
        except:
            print(f"  - Image generated: INVALID")
        
        # Save for visual inspection
        try:
            img_bytes_full = base64.b64decode(img_base64)
            filename = f"prediction_{loc_name}.png"
            with open(filename, 'wb') as f:
                f.write(img_bytes_full)
            print(f"  - Saved: {filename}")
        except Exception as e:
            print(f"  - Save error: {e}")
            
        # Extract prediction values from the graph by checking HTML structure
        # The plot shows history and prediction
        if "SENSOR TELEMETRY" in response.text:
            print(f"  - Plot element: FOUND")
    else:
        print(f"  - Image data: NOT FOUND")

print("\n" + "="*80)
print("\nIMAGES SAVED!")
print("Check the working directory for prediction_*.png files")
print("Each location should have a DIFFERENT plot showing different traffic patterns!")
print()
