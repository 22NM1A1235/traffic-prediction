"""
Extract HTML and check if plot_url is present
"""
import requests

session = requests.Session()
session.post("http://localhost:5000/login", data={"username": "test"})

# Make prediction request
response = session.post("http://localhost:5000/prediction",
                       data={"latitude": "17.4400", "longitude": "78.6100"},
                       allow_redirects=True)

# Check if plot is in response
print("Response length:", len(response.text))
print("HTML contains 'data:image/png;base64':", "data:image/png;base64" in response.text)
print("HTML contains 'plot_url':", "plot_url" in response.text)
print("HTML contains 'selected_sensor':", "selected_sensor" in response.text)
print("HTML contains 'TRAFFIC FLOW FORECAST':", "TRAFFIC FLOW FORECAST" in response.text)

# Check for image tag
if '<img src="data:image/png;base64' in response.text:
    print("\nImage tag found! Plot is in page.")
    # Find the data
    start = response.text.find('<img src="data:image/png;base64')
    end = response.text.find('"', start + 50)
    img_tag = response.text[start:end+50]
    print(f"\nImage tag: {img_tag[:150]}...")
else:
    print("\nImage tag NOT found in response!")
    
    # Check if the conditional failed
    if "No prediction data generated" in response.text:
        print("Template shows 'No prediction data generated' - plot_url not passed?")
    
    # Show what's actually in the results section
    if "SENSOR TELEMETRY" in response.text:
        start = response.text.find("SENSOR TELEMETRY")
        print(f"\nFound data around position {start}")
        print(response.text[start:start+500])

print("\nDone!")
