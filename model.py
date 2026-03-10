import torch
import torch.nn as nn

class TempEncoder(nn.Module):
    """
    Encodes temporal information AND static node features (lat/long).
    MASSIVELY Enhanced: Much deeper static branch for within-state location differentiation
    """
    def __init__(self, input_len, input_dim, static_dim, embed_dim):
        super(TempEncoder, self).__init__()
        # Temporal branch
        self.temporal_mlp = nn.Sequential(
            nn.Linear(input_len * input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # MASSIVELY deeper static branch - much higher capacity for location learning
        # This forces the model to learn fine-grained location-specific patterns
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Fusion layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x, static_feat):
        # x: (Batch, Input_Len, Nodes, Input_Dim)
        # static_feat: (Batch, Nodes, Static_Dim)
        B, T, N, C = x.shape
        
        # Process temporal data: (Batch, Nodes, T*C)
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, T * C)
        temporal_encoded = self.temporal_mlp(x_flat)  # (Batch, Nodes, Embed_Dim)
        
        # MASSIVELY enhanced location encoding - this dominates differentiation
        static_encoded = self.static_mlp(static_feat)  # (Batch, Nodes, Embed_Dim)
        
        # Fuse temporal and spatial information
        combined = torch.cat([temporal_encoded, static_encoded], dim=-1)  # (Batch, Nodes, 2*Embed_Dim)
        fused = self.fusion_mlp(combined)  # (Batch, Nodes, Embed_Dim)
        
        return fused

class STMixerLayer(nn.Module):
    """
    Standard ST-Mixer Layer (Unchanged)
    """
    def __init__(self, num_nodes, embed_dim):
        super(STMixerLayer, self).__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(num_nodes, num_nodes),
            nn.GELU(),
            nn.Linear(num_nodes, num_nodes)
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Spatial Mix
        y = x.permute(0, 2, 1) # (Batch, Embed, Nodes)
        y = self.spatial_mlp(y)
        y = y.permute(0, 2, 1) # (Batch, Nodes, Embed)
        x = self.norm1(x + y)
        
        # Channel Mix
        y = self.channel_mlp(x)
        x = self.norm2(x + y)
        return x

class STMLP(nn.Module):
    def __init__(self, num_nodes, input_len, input_dim, static_dim, embed_dim, output_len, num_layers=3):
        super(STMLP, self).__init__()
        self.num_nodes = num_nodes
        self.static_dim = static_dim
        self.output_len = output_len
        
        self.temp_encoder = TempEncoder(input_len, input_dim, static_dim, embed_dim)
        self.mixers = nn.ModuleList([
            STMixerLayer(num_nodes, embed_dim) for _ in range(num_layers)
        ])
        
        # MASSIVELY enhanced Location encoder - much deeper to force location learning
        # This encodes fine-grained spatial patterns for within-state differentiation
        self.location_encoder = nn.Sequential(
            nn.Linear(static_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_len)
        )
        
        # Base decoder for temporal information
        self.temporal_decoder = nn.Sequential(
            nn.Linear(embed_dim, output_len),
            nn.ReLU()
        )
        
        # Enhanced Fusion decoder
        self.fusion_decoder = nn.Sequential(
            nn.Linear(output_len * 2, output_len * 2),
            nn.ReLU(),
            nn.Linear(output_len * 2, output_len)
        )

    def forward(self, x, static_feat):
        # Temporal encoding
        x_encoded = self.temp_encoder(x, static_feat)  # (Batch, Nodes, Embed_Dim)
        
        # Apply spatial-temporal mixers
        for mixer in self.mixers:
            x_encoded = mixer(x_encoded)  # (Batch, Nodes, Embed_Dim)
        
        # Temporal branch: predict from temporal info
        temporal_out = self.temporal_decoder(x_encoded)  # (Batch, Nodes, Output_Len)
        
        # Location branch: deeply learn location-specific patterns
        # This ensures different coordinates produce different outputs
        location_out = self.location_encoder(static_feat)  # (Batch, Nodes, Output_Len)
        
        # Fuse both branches with LOCATION EMPHASIS
        combined = torch.cat([temporal_out, location_out], dim=-1)  # (Batch, Nodes, Output_Len*2)
        
        # Reshape for fusion decoder to handle 2D input
        B, N, _ = combined.shape
        combined_flat = combined.reshape(B * N, -1)  # (Batch*Nodes, Output_Len*2)
        fused_flat = self.fusion_decoder(combined_flat)  # (Batch*Nodes, Output_Len)
        fused = fused_flat.reshape(B, N, self.output_len)  # (Batch, Nodes, Output_Len)
        
        # CRITICAL: 60% weight toward location output ensures location differences
        # translate to prediction differences even within the same state
        final = fused * 0.4 + location_out * 0.6
        
        return final.permute(0, 2, 1)  # (Batch, Output_Len, Nodes)
