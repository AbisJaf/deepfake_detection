# Run this as a standalone script to check your model's bias
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.detector import MultimodalDeepfakeDetector
# ... rest of script

device = torch.device("cuda")
model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
model.load_state_dict(torch.load(r"D:\fyp\app\deepfake_detection\deepfake_detector_adapted_BEST.pth", map_location=device))
model.eval()

# Feed it pure zeros (no signal at all)
dummy_vis = torch.zeros(1, 16, 3, 224, 224).to(device)
dummy_aud = torch.zeros(1, 1, 80, 300).to(device)

with torch.no_grad():
    logits, _, _ = model(dummy_vis, dummy_aud)
    print("Zero-input logit:", logits.item())
    print("Zero-input sigmoid:", torch.sigmoid(logits).item())