import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Helper class to cleanly build the CNN layers dictated in Table 3.3"""
    def __init__(self, in_channels, out_channels, apply_pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # Only apply MaxPool if specified by the TDD table
        self.pool = nn.MaxPool2d(2, 2) if apply_pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=512, target_frames=16):
        super(AudioEncoder, self).__init__()
        self.target_frames = target_frames
        
        # Input shape will be (Batch, 1 channel, 80 Mels, Time)
        self.layer1 = ConvBlock(1, 64, apply_pool=True)     # Out: (B, 64, 40, T/2)
        self.layer2 = ConvBlock(64, 128, apply_pool=True)   # Out: (B, 128, 20, T/4)
        self.layer3 = ConvBlock(128, 256, apply_pool=False) # Out: (B, 256, 20, T/4)
        self.layer4 = ConvBlock(256, 256, apply_pool=True)  # Out: (B, 256, 10, T/8)
        self.layer5 = ConvBlock(256, 512, apply_pool=False) # Out: (B, 512, 10, T/8)
        self.layer6 = ConvBlock(512, 512, apply_pool=True)  # Out: (B, 512, 5, T/16)
        
        # GAP: Crush the 5 frequency bands into 1, and force the time dimension to exactly 16
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, self.target_frames))
        
        self.projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        # x shape from HDF5: (Batch, 80, 391)
        # CNNs require a "channel" dimension, so we add a dummy channel of 1: (Batch, 1, 80, 391)
        x = x.unsqueeze(1)
        
        # Pass through the CNN-6 block
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        # Shape is currently (Batch, 512, 5, ~24)
        x = self.adaptive_pool(x) 
        # Shape is now (Batch, 512, 1, 16)
        
        # Strip the frequency dimension and swap sequence to match Visual Encoder
        # Output shape: (Batch, 16, 512)
        x = x.squeeze(2).transpose(1, 2)
        
        # Final linear projection
        a_embedding = self.projector(x)
        
        return a_embedding