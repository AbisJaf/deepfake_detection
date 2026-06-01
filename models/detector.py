import torch
import torch.nn as nn
from models.visual_encoder import VisualEncoder
from models.audio_encoder import AudioEncoder
from models.cross_attention import CrossAttentionFusion

class MultimodalDeepfakeDetector(nn.Module):
    def __init__(self, embed_dim=512):
        super(MultimodalDeepfakeDetector, self).__init__()
        
        # Instantiate the three main brain regions
        self.visual_stream = VisualEncoder(embed_dim=embed_dim)
        self.audio_stream = AudioEncoder(embed_dim=embed_dim)
        self.fusion_module = CrossAttentionFusion(embed_dim=embed_dim)
        
        # Classifier Head (Exactly as defined in TDD Section 3.4.2)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid() # Squashes the final number between 0.0 (Real) and 1.0 (Fake)
        )
        
    def forward(self, video_tensor, audio_tensor):
        # 1. Extract Embeddings (Translating raw data into math)
        v_embed = self.visual_stream(video_tensor) 
        a_embed = self.audio_stream(audio_tensor)  
        
        # 2. Cross-Attention Fusion (Checking for desync)
        f = self.fusion_module(v_embed, a_embed)   
        
        # 3. Final Verdict (The probability score)
        probability = self.classifier(f)           
        
        # We return the raw v_embed and a_embed alongside the probability.
        # Why? Because Section 5.1 of your TDD requires them to calculate the LSE-D auxiliary loss!
        return probability, v_embed, a_embed