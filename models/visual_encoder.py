import torch
import torch.nn as nn
from transformers import ViTModel

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super(VisualEncoder, self).__init__()
        
        # Load pre-trained ViT-B/16 from Hugging Face (ImageNet-21k as per TDD Table 3.1)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # PHASE 1 WARM-UP IMPLEMENTATION:
        # The TDD (Section 5.3) dictates freezing the backbone for the first 5 epochs.
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Projection Layer: Reduces ViT's native 768 dimension down to our target 512
        self.projector = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        # Input shape: (Batch, Frames, Channels, Height, Width) -> (B, 16, 3, 224, 224)
        B, T, C, H, W = x.shape
        
        # ViT expects (Batch, Channels, Height, Width). 
        # We merge Batch and Frames to process all 16 frames simultaneously on the GPU.
        x = x.view(B * T, C, H, W)
        
        # Pass through Vision Transformer
        outputs = self.vit(x)
        
        # Extract the [CLS] token (the 0th token) which summarizes the entire frame
        # Shape becomes: (B*T, 768)
        cls_tokens = outputs.last_hidden_state[:, 0, :] 
        
        # Project down to 512 dimensions
        projected = self.projector(cls_tokens)
        
        # Reshape back to a sequence: (Batch, Frames, 512)
        v_embedding = projected.view(B, T, -1)
        
        return v_embedding