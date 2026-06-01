import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepfakeCompositeLoss(nn.Module):
    def __init__(self, lambda_weight=0.3, margin=1.0):
        super(DeepfakeCompositeLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_weight = lambda_weight
        self.margin = margin

    def forward(self, probability, labels, v_embed, a_embed):
        # 1. Primary Loss: Binary Cross Entropy
        bce_loss = self.bce(probability.squeeze(), labels.float())
        
        # 2. Auxiliary Loss: Lip-Synchronisation Error Distance (LSE-D)
        # We average the temporal embeddings across the 16 frames to get a single vector per video
        v_mean = v_embed.mean(dim=1)
        a_mean = a_embed.mean(dim=1)
        
        # Calculate the Euclidean (L2) distance between the visual and audio vectors
        distances = F.pairwise_distance(v_mean, a_mean, p=2)
        
        # If Fake (y=1): Push embeddings apart (maximize distance up to the margin)
        # If Real (y=0): Pull embeddings together (minimize distance)
        lse_d_fake = labels * torch.clamp(self.margin - distances, min=0.0)
        lse_d_real = (1 - labels) * distances
        
        lse_d_loss = (lse_d_fake + lse_d_real).mean()
        
        # 3. Total Loss Combination
        total_loss = bce_loss + (self.lambda_weight * lse_d_loss)
        
        return total_loss, bce_loss, lse_d_loss