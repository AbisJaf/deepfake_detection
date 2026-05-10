import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        # 1. Visual queries Audio (V attends to A)
        # PyTorch's MultiheadAttention uses the exact Q*K^T / sqrt(d_k) math from your TDD
        self.v_attends_a = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 2. Audio queries Visual (A attends to V)
        self.a_attends_v = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 3. Feed Forward Network & LayerNorm for the concatenated vectors
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.ReLU()
        )
        
    def forward(self, v_embed, a_embed):
        # MultiheadAttention expects inputs in the order: (Query, Key, Value)
        
        # Visual looking at Audio
        attn_v2a, _ = self.v_attends_a(v_embed, a_embed, a_embed)
        
        # Audio looking at Visual
        attn_a2v, _ = self.a_attends_v(a_embed, v_embed, v_embed)
        
        # Concatenate them side-by-side
        # Shape goes from two (Batch, 16, 512) tensors to one (Batch, 16, 1024) tensor
        concat_features = torch.cat([attn_v2a, attn_a2v], dim=2)
        
        # Temporal Pooling: We average the 16 time frames down to a single master summary per video
        # Shape becomes: (Batch, 1024)
        pooled_features = torch.mean(concat_features, dim=1)
        
        # Final pass through the Feed Forward Network
        f = self.ffn(pooled_features)
        
        return f