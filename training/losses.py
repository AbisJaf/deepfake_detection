

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepfakeCompositeLoss(nn.Module):
    def __init__(self, lambda_weight=0.3, margin=2.0, pos_weight=0.25):
        super(DeepfakeCompositeLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.margin = margin
        self.pos_weight_val = pos_weight

    def forward(self, probability, labels, v_embed, a_embed):
        # Create BCE loss dynamically on the correct GPU
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight_val], device=labels.device))
        bce_loss = bce(probability.squeeze(), labels.float())
        
        # Average the 16 frames into a single vector
        v_mean = v_embed.mean(dim=1)
        a_mean = a_embed.mean(dim=1)
        
        # --- THE FIX: L2 NORMALIZATION ---
        # Forces the distance to be between 0.0 and 2.0
        v_mean = F.normalize(v_mean, p=2, dim=1)
        a_mean = F.normalize(a_mean, p=2, dim=1)
        # ---------------------------------
        
        distances = F.pairwise_distance(v_mean, a_mean, p=2)
        
        lse_d_fake = labels * torch.clamp(self.margin - distances, min=0.0)
        lse_d_real = (1 - labels) * distances
        lse_d_loss = (lse_d_fake + lse_d_real).mean()
        
        total_loss = bce_loss + (self.lambda_weight * lse_d_loss)
        
        return total_loss, bce_loss, lse_d_loss