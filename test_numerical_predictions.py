"""
Test Flask with different locations and extract prediction numbers
"""
import requests
import re

session = requests.Session()
session.post("http://localhost:5000/login", data={"username": "test"})

# Test locations
locations = [
    ("Cyberabad", 17.4400, 78.6100),
    ("Kukatpally", 17.3821, 78.4184),
    ("HITEC City", 17.3532, 78.4990),
]

print("\nTesting different Hyderabad locations with NUMERICAL PREDICTIONS:\n")
print("="*80 + "\n")

predictions = {}

for loc_name, lat, lon in locations:
    print(f"{loc_name} ({lat:.4f}, {lon:.4f})")
    print("-" * 50)
    
    response = session.post("http://localhost:5000/prediction",
                           data={"latitude": str(lat), "longitude": str(lon)},
                           allow_redirects=True)
    
    # Extract prediction statistics from HTML
    
    # Look for AVERAGE FLOW
    avg_match = re.search(r'AVERAGE FLOW.*?<div[^>]*>(\d+\.\d+)</div>', response.text, re.DOTALL)
    if avg_match:
        avg_val = avg_match.group(1)
        predictions[loc_name] = float(avg_val)
        print(f"  Average Flow: {avg_val} vehicles/hour")
    else:
        print(f"  Could not extract average flow")
        continue
    
    # Look for MAX FLOW
    max_match = re.search(r'MAX FLOW.*?<div[^>]*style="font-size: 1\.8rem[^>]*>(\d+\.\d+)</div>', response.text, re.DOTALL)
    if max_match:
        print(f"  Max Flow: {max_match.group(1)} vehicles/hour")
    
    # Look for MIN FLOW  
    min_match = re.search(r'MIN FLOW.*?<div[^>]*style="font-size: 1\.8rem[^>]*>(\d+\.\d+)</div>', response.text, re.DOTALL)
    if min_match:
        print(f"  Min Flow: {min_match.group(1)} vehicles/hour")
    
    # Check if prediction data is present
    if "LOCATION PREDICTION STATISTICS" in response.text:
        print(f"  Status: PREDICTION GENERATED")
    else:
        print(f"  Status: No prediction stats found")
    
    print()

# Analysis
print("="*80)
if len(predictions) > 1:
    values = list(predictions.values())
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    
    print("\nCOMPARISON:\n")
    for name, val in sorted(predictions.items(), key=lambda x: x[1]):
        print(f"  {name:20} → {val:.2f} vehicles/hour")
    
    print(f"\n  Highest: {max(predictions.items(), key=lambda x: x[1])[0]:20} ({max_val:.2f})")
    print(f"  Lowest:  {min(predictions.items(), key=lambda x: x[1])[0]:20} ({min_val:.2f})")
    print(f"  Difference: {range_val:.2f} vehicles/hour ({(range_val/min_val*100):.1f}% variation)")
    
    if range_val > 0.1:
        print("\n  RESULT: DIFFERENT LOCATIONS PRODUCE DIFFERENT PREDICTIONS!")
    else:
        print("\n  RESULT: Predictions are similar")

print("\n" + "="*80 + "\n")
