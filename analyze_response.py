import re

with open('prediction_response.html', 'r', encoding='utf-8') as f:
    content = f.read()

if 'plot_url' in content:
    print("✓ Found plot_url in response")
    # Find the plot base64 data
    match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]{100})', content)
    if match:
        plot_data = match.group(1)
        print(f"✓ Found plot base64 data: {len(plot_data)} characters")
else:
    print("✗ No plot_url found in response")

if 'Traffic Prediction' in content:
    print("✓ Found 'Traffic Prediction' text")
else:
    print("✗ No 'Traffic Prediction' title found")

if 'selected_sensor' in content:
    # Extract sensor info
    match = re.search(r'selected_sensor["\']?\s*:\s*([^,}]+)', content)
    if match:
        print(f"  Sensor info: {match.group(1)[:100]}")

# Print last 500 chars to see closing
print("\nLast 500 chars of file:")
print(content[-500:])
