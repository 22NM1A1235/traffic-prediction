import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from model import STMLP

# --- Configuration ---
INPUT_LEN = 12       # Past 12 steps (1 hour)
OUTPUT_LEN = 12      # Future 12 steps (1 hour)
INPUT_DIM = 2        # CHANGED: Now using 2 features (Flow + Hour of Day)
STATIC_DIM = 2       # Latitude, Longitude
EMBED_DIM = 64
EPOCHS = 2           # Very short for fast training with enhanced learning  
BATCH_SIZE = 256     # Larger batch for faster training with all sensors
REG_LAMBDA = 0       # CHANGED: Set to 0 to prevent over-smoothing

# FIXED: Use single-node model for per-location predictions
# Set to False to train multi-node model (trained on all sensors together)
# Set to True to train single-node model (trained per sensor, better for location-specific predictions)
SINGLE_NODE_MODEL = True

def load_data():
    print("Loading and processing datasets...")
    
    # 1. Load Traffic Series (Dynamic Features)
    df_ts = pd.read_csv('traffic_time_series.csv')
    
    # Feature Engineering: Add Normalized Hour (0-1)
    df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
    df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0
    
    # Pivot Flow: (Time x Sensors)
    pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
    # Pivot Hour: (Time x Sensors) - repeats for all sensors
    pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()
    
    # Ensure consistent sensor order
    # FIXED: Use ALL sensors including AP regions - don't limit to 200
    sensor_ids = sorted(pivot_flow.columns)
    print(f"Training with {len(sensor_ids)} sensors (Bengaluru + Andhra Pradesh)")
    pivot_flow = pivot_flow[sensor_ids]
    pivot_hour = pivot_hour[sensor_ids]
    
    # Normalize Flow
    scaler = StandardScaler()
    flow_values = scaler.fit_transform(pivot_flow.values)
    hour_values = pivot_hour.values # Already 0-1
    
    # Combine into (Time, Nodes, Features=2)
    # Stack along last axis
    data_combined = np.stack([flow_values, hour_values], axis=-1)
    
    # 2. Load Sensors Metadata (Static Features)
    df_sensors = pd.read_csv('sensors.csv')
    df_sensors = df_sensors.set_index('sensor_id').reindex(sensor_ids).reset_index()
    static_feats = df_sensors[['latitude', 'longitude']].values
    static_scaler = StandardScaler()
    static_feats_norm = static_scaler.fit_transform(static_feats)
    
    # 3. Load Adjacency Edges (Graph Structure)
    # We load this to compute Laplacian, even if REG_LAMBDA is 0 (for future use)
    df_edges = pd.read_csv('adjacency_edges.csv')
    adj_matrix = np.zeros((len(sensor_ids), len(sensor_ids)))
    sensor_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}
    
    for _, row in df_edges.iterrows():
        if row['source_sensor'] in sensor_to_idx and row['target_sensor'] in sensor_to_idx:
            i, j = sensor_to_idx[row['source_sensor']], sensor_to_idx[row['target_sensor']]
            adj_matrix[i, j] = row['connection_weight']
            adj_matrix[j, i] = row['connection_weight']
            
    # Compute Laplacian
    degree = np.sum(adj_matrix, axis=1)
    # Avoid division by zero
    degree[degree == 0] = 1e-5 
    laplacian = np.diag(degree) - adj_matrix
    d_inv_sqrt = np.diag(np.power(degree, -0.5))
    norm_laplacian = np.dot(np.dot(d_inv_sqrt, laplacian), d_inv_sqrt)
    
    # 4. Load Sequences (Train Splits)
    df_seq = pd.read_csv('traffic_sequences.csv')
    train_indices = df_seq[df_seq['dataset_split'] == 'train']['history_start_step'].unique()
    
    # Create Batches
    def create_dataset(indices, single_node=False):
        """
        Create dataset for training.
        If single_node=True: Create per-sensor datasets (each sample is one sensor's time series)
        If single_node=False: Create multi-node datasets (each sample includes all sensors)
        """
        X, Y = [], []
        
        if single_node:
            # Create separate training samples for EACH sensor individually
            # This ensures the model learns location-specific patterns
            for sensor_idx in range(len(sensor_ids)):
                sensor_flow = flow_values[:, sensor_idx]  # Single sensor's flow over time
                
                for idx in indices:
                    if idx + INPUT_LEN + OUTPUT_LEN < len(sensor_flow):
                        # Input: Single sensor's past flow + hour features
                        # Shape: (INPUT_LEN, 2) for single sensor
                        past_flow = sensor_flow[idx : idx + INPUT_LEN].reshape(-1, 1)  # (INPUT_LEN, 1)
                        past_hour = hour_values[idx : idx + INPUT_LEN, sensor_idx].reshape(-1, 1)  # (INPUT_LEN, 1)
                        sample_input = np.hstack([past_flow, past_hour])  # (INPUT_LEN, 2)
                        X.append(sample_input)
                        
                        # Target: Single sensor's future flow only
                        future_flow = sensor_flow[idx + INPUT_LEN : idx + INPUT_LEN + OUTPUT_LEN]  # (OUTPUT_LEN,)
                        Y.append(future_flow)
        else:
            # Original multi-node approach: all sensors in one input
            for idx in indices:
                if idx + INPUT_LEN + OUTPUT_LEN < len(data_combined):
                    X.append(data_combined[idx : idx + INPUT_LEN])
                    Y.append(data_combined[idx + INPUT_LEN : idx + INPUT_LEN + OUTPUT_LEN, :, 0])
        
        return np.array(X), np.array(Y)

    train_X, train_Y = create_dataset(train_indices, single_node=SINGLE_NODE_MODEL)
    
    print(f"Training Samples: {train_X.shape[0]}")
    print(f"Input Shape: {train_X.shape}")
    print(f"Target Shape: {train_Y.shape}")
    
    # FIXED: Reshape data and static features based on model mode
    if SINGLE_NODE_MODEL:
        # For single-node model:
        # - Input: (num_samples, INPUT_LEN, 2) -> reshape to (num_samples, INPUT_LEN, 1, 2)
        # - Target: (num_samples, OUTPUT_LEN) -> reshape to (num_samples, OUTPUT_LEN, 1)
        # - Static: Create per-sample static features for each sensor
        train_X = train_X.reshape(train_X.shape[0], INPUT_LEN, 1, 2)
        train_Y = train_Y.reshape(train_Y.shape[0], OUTPUT_LEN, 1)
        
        # Create static features that repeat for each sample
        # If we have N sensors, we have N*len(train_indices) samples
        # Static features should cycle through each sensor's coordinates
        num_sensor_samples_each = train_X.shape[0] // len(sensor_ids)
        static_feats_expanded = []
        for sensor_idx in range(len(sensor_ids)):
            sensor_static = static_feats_norm[sensor_idx:sensor_idx+1]  # (1, 2)
            # Repeat this sensor's static features for all its samples
            sensor_static_repeated = np.tile(sensor_static, (num_sensor_samples_each, 1, 1))  # (num_samples, 1, 2)
            static_feats_expanded.append(sensor_static_repeated)
        static_feats_expanded = np.vstack(static_feats_expanded)  # (total_samples, 1, 2)
        print(f"Reshaped for Single-Node Mode:")
        print(f"  Input: {train_X.shape} (Batch, Time, Nodes=1, Features)")
        print(f"  Target: {train_Y.shape} (Batch, Output_Len, Nodes=1)")
        print(f"  Static: {static_feats_expanded.shape} (Batch, Nodes=1, Static_Dim)")
    else:
        # For multi-node model: keep original shapes
        # Input: (num_samples, INPUT_LEN, num_sensors, 2)
        # Target: (num_samples, OUTPUT_LEN, num_sensors)
        # Static: (num_sensors, 2) - same for all samples
        static_feats_expanded = static_feats_norm
        print(f"Using Multi-Node Mode:")
        print(f"  Input: {train_X.shape} (Batch, Time, Nodes, Features)")
        print(f"  Target: {train_Y.shape} (Batch, Output_Len, Nodes)")
    
    return {
        'train_x': torch.tensor(train_X, dtype=torch.float32),
        'train_y': torch.tensor(train_Y, dtype=torch.float32),
        'static_feat': torch.tensor(static_feats_expanded, dtype=torch.float32),
        'static_feat_base': torch.tensor(static_feats_norm, dtype=torch.float32),  # Original for reference
        'laplacian': torch.tensor(norm_laplacian, dtype=torch.float32),
        'num_nodes': len(sensor_ids),
        'scaler': scaler,
        'static_scaler': static_scaler,
        'sensor_ids': sensor_ids,
        'single_node_mode': SINGLE_NODE_MODEL
    }

def train_model():
    data = load_data()
    
    # Save Scalers for App
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/scaler.pkl', 'wb') as f:
        pickle.dump(data['scaler'], f)
    with open('saved_models/static_scaler.pkl', 'wb') as f:
        pickle.dump(data['static_scaler'], f)
    with open('saved_models/sensor_ids.pkl', 'wb') as f:
        pickle.dump(data['sensor_ids'], f)

    # Initialize Model
    # FIXED: Use num_nodes=1 for single-node model (location-specific predictions)
    # Use num_nodes=len(sensor_ids) for multi-node model (global predictions)
    num_nodes = 1 if SINGLE_NODE_MODEL else data['num_nodes']
    
    print(f"Training Model with num_nodes={num_nodes} ({'Single Node' if SINGLE_NODE_MODEL else 'Multi Node'} Mode)")
    
    model = STMLP(
        num_nodes=num_nodes,
        input_len=INPUT_LEN,
        input_dim=INPUT_DIM, 
        static_dim=STATIC_DIM,
        embed_dim=EMBED_DIM,
        output_len=OUTPUT_LEN
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Prepare data tensors
    static_feat_base = data['static_feat']  # This is either per-sample (single-node) or global (multi-node)
    laplacian = data['laplacian']
    train_x = data['train_x']
    train_y = data['train_y']
    single_node_mode = data['single_node_mode']
    
    print(f"Starting Training for {EPOCHS} epochs...")
    model.train()
    
    num_samples = len(train_x)
    
    # ADDED: Coordinate augmentation noise - helps model differentiate based on location
    # MASSIVELY INCREASED to 0.5 to force rapid location differentiation learning
    # Higher noise = model learns to be sensitive to coordinate variations
    # This is KEY to getting within-state location differentiation
    coord_aug_std = 0.5  # Massively increased from 0.35 for faster learning
    
    for epoch in range(EPOCHS):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0
        
        for i in range(0, num_samples, BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            
            # ENHANCED: Handle static features with coordinate augmentation
            curr_batch = batch_x.size(0)
            if single_node_mode:
                # For single-node: static_feat_base has shape (num_samples, 1, 2)
                # Extract the static features for this batch
                batch_static = static_feat_base[indices].clone()  # (curr_batch, 1, 2)
                
                # ADDED: Add small random perturbations to coordinates during training
                # This teaches the model that nearby coordinates have related but different outputs
                if epoch < EPOCHS * 0.8:  # Apply augmentation for first 80% of training
                    coord_noise = torch.randn_like(batch_static) * coord_aug_std
                    batch_static = batch_static + coord_noise
                    # Clamp to reasonable ranges to avoid extreme values
                    batch_static = torch.clamp(batch_static, -3.0, 3.0)
            else:
                # For multi-node: static_feat_base has shape (num_nodes, 2)
                # Expand to (curr_batch, num_nodes, 2)
                batch_static = static_feat_base.unsqueeze(0).expand(curr_batch, -1, -1).clone()
                
                # Apply augmentation to multi-node mode as well
                if epoch < EPOCHS * 0.8:
                    coord_noise = torch.randn_like(batch_static) * coord_aug_std
                    batch_static = batch_static + coord_noise
                    batch_static = torch.clamp(batch_static, -3.0, 3.0)
            
            optimizer.zero_grad()
            
            # Forward Pass
            output = model(batch_x, batch_static) # Output is (Batch, Output_Len, Nodes)
            
            # Loss Calculation
            mse_loss = criterion(output, batch_y)
            
            # Optional: Graph Regularization (Currently Disabled via REG_LAMBDA=0)
            if REG_LAMBDA > 0:
                pred_mean = output.mean(dim=[0, 1])
                reg_loss = torch.matmul(pred_mean.unsqueeze(0), torch.matmul(laplacian, pred_mean.unsqueeze(1)))
                loss = mse_loss + (REG_LAMBDA * reg_loss.squeeze())
            else:
                loss = mse_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 1 == 0:
            avg_loss = epoch_loss / (num_samples / BATCH_SIZE)
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), 'saved_models/st_mlp.pth')
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    train_model()