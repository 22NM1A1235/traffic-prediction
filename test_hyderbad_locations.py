import torch
import pickle
import numpy as np
from model import STMLP

try:
    print("Loading model and scalers...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = STMLP(num_nodes=1, input_len=12, input_dim=2, static_dim=2, embed_dim=64, output_len=12)
    print("Model created")
    
    state = torch.load('saved_models/st_mlp.pth', map_location=device)
    print(f"State dict keys: {list(state.keys())[:5]}...")
    model.load_state_dict(state)
    model.eval()
    print("Model loaded")

    with open('saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('saved_models/static_scaler.pkl', 'rb') as f:
        static_scaler = pickle.load(f)
    print("Scalers loaded")

    print("=" * 80)
    print("TESTING DIFFERENT HYDERABAD LOCATIONS")
    print("=" * 80)

    # Different Hyderabad locations (lat, lon)
    locations = {
        "Location A (Banjara Hills)": (17.3850, 78.4867),
        "Location B (Gachibowli)": (17.3589, 78.5941),
        "Location C (Kukatpally)": (17.3689, 78.3800),
        "Location D (HITEC City)": (17.3595, 78.5889),
    }

    # Sample traffic sequence (normalized) - (Batch=1, Time=12, Nodes=1, Features=2)
    dummy_sequence = np.random.randn(12, 2) * 0.1
    traffic_input = torch.FloatTensor(dummy_sequence).unsqueeze(0).unsqueeze(2)  # (1, 12, 1, 2)
    print(f"Traffic input shape: {traffic_input.shape}")

    predictions = {}

    with torch.no_grad():
        for location_name, (lat, lon) in locations.items():
            # Normalize coordinates
            norm_coords = static_scaler.transform([[lat, lon]])  # Shape: (1, 2)
            coords = torch.FloatTensor(norm_coords)  # Shape: (1, 2)
            coords = coords.unsqueeze(0)  # Shape: (1, 1, 2) - Batch, Nodes, Static_Dim
            
            # Get prediction
            output = model(traffic_input, coords)  # (1, 12, 1)
            # Average the 12 predictions
            pred_value = output.mean().item()
            
            predictions[location_name] = {
                'coords': (lat, lon),
                'prediction': pred_value,
                'normalized_coords': norm_coords[0]
            }
            
            print(f"\n{location_name}")
            print(f"  Coordinates: {lat:.4f}, {lon:.4f}")
            print(f"  Normalized: {norm_coords[0]}")
            print(f"  Prediction (mean): {pred_value:.4f}")

    print("\n" + "=" * 80)
    print("PREDICTION DIFFERENCES:")
    print("=" * 80)

    pred_values = [p['prediction'] for p in predictions.values()]
    min_pred = min(pred_values)
    max_pred = max(pred_values)
    range_pred = max_pred - min_pred

    print(f"Min prediction: {min_pred:.4f}")
    print(f"Max prediction: {max_pred:.4f}")
    print(f"Range: {range_pred:.4f}")
    print(f"Std Dev: {np.std(pred_values):.4f}")
    if np.mean(pred_values) != 0:
        print(f"Coefficient of Variation: {np.std(pred_values) / abs(np.mean(pred_values)) * 100:.2f}%")

    if range_pred < 0.001:
        print("\n⚠️  ISSUE CONFIRMED: Predictions are identical/nearly identical!")
        print("    The location encoder is NOT being used properly.")
    else:
        print(f"\n✅ Location sensitivity OK: {range_pred:.4f} variation detected")
        
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
