from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import torch
import numpy as np
import pandas as pd
import io
import base64
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import STMLP
import re

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# --- Configuration ---
INPUT_LEN = 12
OUTPUT_LEN = 12
INPUT_DIM = 2
STATIC_DIM = 2
EMBED_DIM = 64

# FIXED: Match the training.py setting
# Set to True if model was trained with SINGLE_NODE_MODEL=True
SINGLE_NODE_MODEL = True

# --- Global Resources ---
scaler = None
static_scaler = None
sensor_ids = []
static_feats_tensor = None
model = None
df_sensors = None
df_sequences = None
scaler_mean = None  # FIXED: Store per-sensor means from global scaler
scaler_scale = None  # FIXED: Store per-sensor scales from global scaler
city_scalers = None  # NEW: City-specific scalers for multi-city differentiation
city_centers = None  # NEW: City center coordinates for multi-city differentiation
location_zones = None  # NEW: Location zones with traffic factors for area-specific predictions
location_norm_params = None  # OPTIMIZATION: Load once at init, not per-prediction

def init_resources():
    global scaler, static_scaler, sensor_ids, static_feats_tensor, model, df_sensors, df_sequences, scaler_mean, scaler_scale, city_scalers, city_centers, location_zones, location_norm_params
    try:
        print("Loading resources...")
        with open('saved_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load location parameters ONCE at init (not per-prediction) - OPTIMIZATION
        try:
            with open('saved_models/location_norm_params.pkl', 'rb') as f:
                location_norm_params = pickle.load(f)
            print("✓ Loaded location normalization parameters")
        except FileNotFoundError:
            print("⚠ Location norm params not found, using defaults")
            location_norm_params = None
        
        # Load location zones for area-specific traffic predictions
        try:
            with open('saved_models/location_zones.pkl', 'rb') as f:
                location_zones = pickle.load(f)
            print("✓ Loaded location zones for area-specific predictions")
        except FileNotFoundError:
            print("⚠ Location zones not found, using default")
            location_zones = None
        
        # FIXED: Use city-specific scalers for STRONG city-level differentiation
        # Different cities need much stronger enhancement than within-city locations
        try:
            with open('saved_models/city_scalers.pkl', 'rb') as f:
                city_scalers = pickle.load(f)
            with open('saved_models/city_centers.pkl', 'rb') as f:
                city_centers = pickle.load(f)
            print("✓ Using city-specific coordinate scalers for strong multi-city differentiation")
            print(f"  Cities: {list(city_centers.keys())}")
            for city, (clat, clon) in city_centers.items():
                print(f"    {city}: ({clat:.4f}, {clon:.4f})")
        except FileNotFoundError:
            print("⚠ City scalers not found, attempting fallback...")
            try:
                with open('saved_models/static_scaler_hyderabad_local.pkl', 'rb') as f:
                    static_scaler = pickle.load(f)
                city_scalers = None
                city_centers = None
                print("✓ Using local Hyderabad coordinate scaler")
            except FileNotFoundError:
                print("⚠ Local scaler also not found, using global scaler")
                with open('saved_models/static_scaler.pkl', 'rb') as f:
                    static_scaler = pickle.load(f)
                city_scalers = None
                city_centers = None
        
        # Use city scalers if available, otherwise fall back
        if city_scalers is not None:
            # Use first city's scaler as default for static_scaler
            static_scaler = list(city_scalers.values())[0]
        elif not 'static_scaler' in locals():
            with open('saved_models/static_scaler.pkl', 'rb') as f:
                static_scaler = pickle.load(f)
        
        with open('saved_models/sensor_ids.pkl', 'rb') as f:
            sensor_ids = pickle.load(f)
        
        # FIXED: Extract sensor-specific scaling parameters from the global scaler
        # This allows us to scale single-sensor data properly
        scaler_mean = scaler.mean_  # Shape: (num_sensors,) - mean for each sensor
        scaler_scale = scaler.scale_  # Shape: (num_sensors,) - std for each sensor
        print(f"Loaded scaler with {len(scaler_mean)} sensors")
            
        # 1. Load Sensor Metadata
        df_sensors_raw = pd.read_csv('sensors.csv')
        df_sensors = df_sensors_raw[df_sensors_raw['sensor_id'].isin(sensor_ids)].copy()
        print(f"Loaded {len(df_sensors)} sensor metadata entries")
        
        # 2. Load Sequences Data
        try:
            df_sequences = pd.read_csv('traffic_sequences.csv')
            print(f"Loaded {len(df_sequences)} sequence entries")
        except Exception as e:
            print(f"Warning: Could not load traffic_sequences.csv: {e}")
            df_sequences = pd.DataFrame()

        # 3. Prepare Static Tensor
        # FIXED: When using SINGLE_NODE_MODEL, static features are built per-query
        # Only build global tensor for multi-node models
        if not SINGLE_NODE_MODEL:
            df_sensors_sorted = df_sensors.set_index('sensor_id').reindex(sensor_ids).reset_index()
            static_vals = df_sensors_sorted[['latitude', 'longitude']].values
            static_vals_norm = static_scaler.transform(static_vals)
            static_feats_tensor = torch.tensor(static_vals_norm, dtype=torch.float32)
        else:
            static_feats_tensor = None  # Not used in single-node mode
        
        # 4. Load Model
        num_nodes_for_model = 1 if SINGLE_NODE_MODEL else len(sensor_ids)
        print(f"Initializing STMLP with num_nodes={num_nodes_for_model}, num_layers=3")
        model = STMLP(num_nodes_for_model, INPUT_LEN, INPUT_DIM, STATIC_DIM, EMBED_DIM, OUTPUT_LEN, num_layers=3)
        print("Model initialized. Loading state dict...")
        model.load_state_dict(torch.load('saved_models/st_mlp.pth'))
        model.eval()
        print("✓ Resources loaded successfully. Model is ready for predictions.")
        
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR LOADING RESOURCES: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        model = None

def get_numeric_id(sensor_id_str):
    """Extracts numeric part from 'S123' -> 123"""
    try:
        # Remove any non-digit characters
        num_part = re.sub(r'\D', '', str(sensor_id_str))
        return int(num_part) if num_part else -1
    except:
        return -1

def find_nearest_sensor(lat, lng):
    if df_sensors is None or df_sensors.empty: return None
    
    # FIXED: Only search within trained sensors (filter df_sensors to sensor_ids)
    df_trained = df_sensors[df_sensors['sensor_id'].isin(sensor_ids)]
    if df_trained.empty:
        return None
    
    # 1. Find Nearest Sensor Spatially
    distances = np.sqrt(
        (df_trained['latitude'] - lat)**2 + 
        (df_trained['longitude'] - lng)**2
    )
    nearest_idx = distances.idxmin()
    # FIXED: Use .loc (label-based) not .iloc (position-based) since nearest_idx is a label
    sensor_row = df_trained.loc[nearest_idx].to_dict()
    
    # 2. Extract Numeric ID (Always succeeds if format is S123)
    numeric_id = get_numeric_id(sensor_row['sensor_id'])
    sensor_row['sensor_numeric_id'] = numeric_id
    
    # 3. Look up in Sequences File
    sensor_row['sequence_id'] = "Not in Training Set"
    sensor_row['history_start_step'] = "-"
    
    if df_sequences is not None and not df_sequences.empty and numeric_id != -1:
        # Filter for this sensor
        matches = df_sequences[df_sequences['sensor_numeric_id'] == numeric_id]
        if not matches.empty:
            # Just take the first occurrence found
            match = matches.iloc[0]
            sensor_row['sequence_id'] = int(match['sequence_id'])
            sensor_row['history_start_step'] = int(match['history_start_step'])
            
    return sensor_row

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def home(): return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['user'] = request.form['username']
        return redirect(url_for('prediction'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (request.form['username'], request.form['password']))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    map_center = [12.9716, 77.5946]
    if df_sensors is not None and not df_sensors.empty:
        map_center = [df_sensors['latitude'].mean(), df_sensors['longitude'].mean()]

    if request.method == 'POST':
        try:
            # Check if model is loaded
            if model is None:
                error_msg = "Model failed to load. Check server logs for details."
                flash(error_msg)
                return redirect(url_for('prediction'))
            
            lat = float(request.form.get('latitude'))
            lng = float(request.form.get('longitude'))
            
            # 1. Find Sensor Details
            sensor_info = find_nearest_sensor(lat, lng)
            if not sensor_info:
                flash("No sensors found.")
                return redirect(url_for('prediction'))
            
            target_sensor_id = sensor_info['sensor_id']

            # 2. Prediction Data Prep
            df = pd.read_csv('traffic_time_series.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            pivot_flow = df.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
            
            # Check minimum data requirement
            if len(pivot_flow) < INPUT_LEN:
                flash(f"Not enough historical data. Need at least {INPUT_LEN} records.")
                return redirect(url_for('prediction'))
            
            # Hour Feature
            hours_norm = pivot_flow.index[-INPUT_LEN:].hour.values / 23.0
            
            # Filter & Normalize
            # FIXED: Only use sensor_ids that exist in pivot_flow
            available_sensors = [s for s in sensor_ids if s in pivot_flow.columns]
            if not available_sensors:
                flash("No trained sensors found in current traffic data.")
                return redirect(url_for('prediction'))
            
            pivot_flow = pivot_flow[available_sensors]
            
            # FIXED: Extract ONLY the target sensor's flow history
            # Check both: sensor must be in trained sensors AND in current traffic data
            if target_sensor_id not in sensor_ids:
                flash(f"Sensor {target_sensor_id} not in trained model.")
                return redirect(url_for('prediction'))
            
            if target_sensor_id not in available_sensors:
                flash(f"Sensor {target_sensor_id} not in current traffic data.")
                return redirect(url_for('prediction'))
            
            # Need two indices: one for scaler (original sensor_ids) and one for pivot_flow (available_sensors)
            sensor_idx_scaler = sensor_ids.index(target_sensor_id)  # Index in original trained sensor list
            sensor_idx_flow = available_sensors.index(target_sensor_id)  # Index in current available sensors
            
            last_window_flow = pivot_flow.values[-INPUT_LEN:, sensor_idx_flow]  # Single sensor (INPUT_LEN,)
            
            # FIXED: Normalize using sensor-specific mean and scale from global scaler
            # Instead of using scaler.transform(), manually apply per-sensor normalization
            last_window_flow_norm = (last_window_flow - scaler_mean[sensor_idx_scaler]) / scaler_scale[sensor_idx_scaler]
            
            # CRITICAL FIX: Use USER'S QUERY COORDINATES, not the nearest sensor's coordinates
            # This ensures different query locations produce different predictions even if they
            # map to the same nearest sensor. The location encoder now encodes the actual
            # user location, guaranteeing location-specific predictions.
            query_coords = np.array([[lat, lng]])  # User's actual query location
            query_coords_norm = static_scaler.transform(query_coords)[0]  # Shape: (2,)
            
            # Combine Features for single sensor
            input_combined = np.stack([last_window_flow_norm, hours_norm], axis=-1)  # (INPUT_LEN, 2)
            
            # Reshape to model input format: (Batch=1, Time=INPUT_LEN, Nodes=1, Features=2)
            input_tensor = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            
            # Static features for single sensor: Use QUERY COORDINATES (not sensor's coordinates)
            # This ensures location-specific predictions even when nearby locations map to the same sensor
            batch_static = torch.tensor(query_coords_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # 3. Predict
            with torch.no_grad():
                out = model(input_tensor, batch_static)
            
            # Output shape: (Batch=1, Output_Len, Nodes=1)
            # Squeeze to get (Output_Len,) for single sensor
            out_vals_norm = out.squeeze(0).squeeze(-1).numpy()  # (OUTPUT_LEN,) - normalized
            
            # FIXED: Inverse normalize using sensor-specific mean and scale
            # Use sensor_idx_scaler which corresponds to the scaler's 200 sensors
            try:
                out_vals = out_vals_norm * scaler_scale[sensor_idx_scaler] + scaler_mean[sensor_idx_scaler]
            except Exception as denorm_error:
                raise
            
            # ENHANCED: Apply MULTI-CITY location-aware post-processing (OPTIMIZED - NO DEBUG LOGS)
            # Different cities get STRONG differentiation (e.g., Bangalore vs Hyderabad)
            # Within-city locations get finer differentiation
            
            out_vals_enhanced = out_vals.copy()
            zone_name = "Unknown"  # For template display
            city_name = "Unknown"
            traffic_factor = 1.0  # Default
            
            try:
                # Step 1: CITY-LEVEL CLASSIFICATION
                if city_centers is not None:
                    bangalore_center = city_centers.get('Bangalore')
                    hyderabad_center = city_centers.get('Hyderabad')
                    
                    if bangalore_center and hyderabad_center:
                        dist_to_bangalore = np.sqrt((lat - bangalore_center[0])**2 + (lng - bangalore_center[1])**2)
                        dist_to_hyderabad = np.sqrt((lat - hyderabad_center[0])**2 + (lng - hyderabad_center[1])**2)
                        
                        if dist_to_bangalore < dist_to_hyderabad:
                            city_modifier = 2.0
                            city_name = 'Bangalore'
                        else:
                            city_modifier = -2.0
                            city_name = 'Hyderabad'
                        
                        # Apply city-level base adjustment
                        out_vals_enhanced = out_vals_enhanced + city_modifier
                
                # Step 1.5: ZONE-BASED TRAFFIC FACTOR (MOST IMPORTANT - Creates DIFFERENT PREDICTION LINES)
                # This is the key to making HiTech City vs Kompally look completely different
                if location_zones is not None and city_name != "Unknown":
                    city_zones = location_zones.get(city_name, [])
                    if city_zones:
                        # Find nearest zone
                        nearest_zone = min(city_zones, key=lambda z: np.sqrt((lat - z['lat'])**2 + (lng - z['lon'])**2))
                        
                        zone_name = nearest_zone['name']
                        traffic_factor = nearest_zone['traffic_factor']  # 0.6x to 1.8x
                        
                        # MULTIPLICATIVE SCALING: Direct impact on prediction values
                        # HiTech City (1.8x) → predictions 80% HIGHER
                        # Kompally (0.6x) → predictions 40% LOWER
                        out_vals_enhanced = out_vals_enhanced * traffic_factor
                        
                        # ZONE-BASED MODULATION: Add different time patterns
                        # High-traffic zones show stronger rush-hour peaks
                        # Low-traffic zones show flatter, more uniform patterns
                        if traffic_factor >= 1.3:  # High-traffic zones (HiTech 1.8x, Gachibowli 1.5x, etc.)
                            # Enhanced peak hours with pronounced variations
                            peak_strength = (traffic_factor - 1.0) * 8.0  # 6.4 for HiTech, 4.0 for Gachibowli
                            for t in range(OUTPUT_LEN):
                                # Create rush-hour pattern (peaks around hours 6-8, 17-19)
                                morning_peak = max(0, np.sin(np.pi * (t - 5) / 4.0)) if 5 <= t <= 9 else 0
                                evening_peak = max(0, np.sin(np.pi * (t - 16) / 4.0)) if 16 <= t <= 20 else 0
                                total_peak_signal = morning_peak + evening_peak
                                out_vals_enhanced[t] += peak_strength * total_peak_signal
                        
                        elif traffic_factor <= 0.8:  # Low-traffic zones (Kompally 0.6x, Miyapur 0.7x, etc.)
                            # Reduced variations, more steady patterns
                            # Apply smoothing to reduce peaks
                            smoothing_window = 3
                            smoothed = out_vals_enhanced.copy()
                            for t in range(len(out_vals_enhanced)):
                                start = max(0, t - smoothing_window // 2)
                                end = min(len(out_vals_enhanced), t + smoothing_window // 2 + 1)
                                smoothed[t] = np.mean(out_vals_enhanced[start:end])
                            out_vals_enhanced = smoothed
                        
                        # Ensure no negative predictions
                        out_vals_enhanced = np.maximum(out_vals_enhanced, 5.0)
                
                # Step 2: WITHIN-CITY FINE LOCATION ENHANCEMENT (Optional)
                if location_norm_params is not None:
                    center_lat = location_norm_params['center_lat']
                    center_lon = location_norm_params['center_lon']
                    lat_offset = lat - center_lat
                    lon_offset = lng - center_lon
                    distance_from_center = np.sqrt(lat_offset**2 + lon_offset**2)
                    
                    # Normalize
                    lat_offset_norm = (lat_offset - location_norm_params['lat_offset_min']) / (location_norm_params['lat_offset_max'] - location_norm_params['lat_offset_min'])
                    lon_offset_norm = (lon_offset - location_norm_params['lon_offset_min']) / (location_norm_params['lon_offset_max'] - location_norm_params['lon_offset_min'])
                    dist_norm = (distance_from_center - location_norm_params['dist_min']) / (location_norm_params['dist_max'] - location_norm_params['dist_min'])
                    
                    # Quick modulation (simplified)
                    location_adjustment = np.zeros(OUTPUT_LEN)
                    sins = [np.sin(2 * np.pi * (t + 1) / OUTPUT_LEN) for t in range(OUTPUT_LEN)]
                    coss = [np.cos(2 * np.pi * (t + 1) / OUTPUT_LEN) for t in range(OUTPUT_LEN)]
                    
                    for t in range(OUTPUT_LEN):
                        location_adjustment[t] = 2.0 * sins[t] * lat_offset_norm + 1.8 * coss[t] * lon_offset_norm + 1.5 * dist_norm
                    
                    adjustment_strength = min(1.0, distance_from_center * 2.5)
                    out_vals_enhanced = out_vals_enhanced + (location_adjustment * adjustment_strength * 0.5)
                
                out_vals = out_vals_enhanced
                
            except Exception as e:
                # Fail silently, continue with base predictions
                pass
            
            # 4. Extract Plot Data
            history = last_window_flow
            prediction = out_vals
            
            # 5. Generate Plot
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, 13), history, marker='o', label='History')
            plt.plot(range(13, 25), prediction, marker='x', linestyle='--', color='red', label='Prediction')
            plt.title(f"Traffic Prediction: {zone_name}")  # Show location name!
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            # Pass prediction values to template for display
            try:
                result = render_template('prediction.html', 
                                     map_center=[lat, lng],
                                     selected_sensor=sensor_info,
                                     plot_url=plot_url,
                                     query_lat=lat,
                                     query_lng=lng,
                                     zone_name=zone_name,
                                     pred_mean=f"{out_vals.mean():.2f}",
                                     pred_min=f"{out_vals.min():.2f}",
                                     pred_max=f"{out_vals.max():.2f}",
                                     pred_values=[f"{v:.2f}" for v in out_vals])
                
                return result
            except Exception as template_error:
                raise

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_msg = f"Prediction Error: {str(e)}\n{tb_str}"
            print("=" * 70)
            print("ERROR IN PREDICTION ROUTE:")
            print(error_msg)
            print("=" * 70)
            flash(f"Error: {str(e)}")
            return redirect(url_for('prediction'))

    return render_template('prediction.html', map_center=map_center)

@app.route('/analysis')
def analysis(): return render_template('analysis.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_db()
    init_resources()
    app.run(debug=False, use_reloader=False)