import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detector import MultimodalDeepfakeDetector

class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.visual_stream.vit.encoder.layer[-2]
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        # Fix: PyTorch requires full_backward_hook for proper gradient extraction
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, visual_input, audio_input):
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        logits, v_embed, a_embed = self.model(visual_input, audio_input)
        
        # --- THE MASTERSTROKE: BACKPROPAGATE THE DISTANCE, NOT THE LOGITS ---
        # We average the embeddings just like we do in the loss function
        v_mean = v_embed.mean(dim=1)
        a_mean = a_embed.mean(dim=1)
        
        # Calculate the distance between the Audio and the Video
        distance = F.pairwise_distance(v_mean, a_mean, p=2)
        
        # Backpropagate the distance itself. 
        # This physically forces Grad-CAM to map only the pixels that tie to the audio sync.
        distance.backward()
        # --------------------------------------------------------------------
        
        if self.gradients is None:
            raise ValueError("Gradients are empty. Ensure the ViT backbone is unfrozen.")
            
        acts = self.activations[0, 1:, :]  # Shape: (196, 768)
        grads = self.gradients[0, 1:, :]   # Shape: (196, 768)
        
        weights = torch.mean(grads, dim=0) # Shape: (768,)
        cam = torch.matmul(acts, weights)  # Shape: (196,)
        
        cam = cam.view(14, 14)
        cam = torch.abs(cam)
        
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam.cpu().detach().numpy()

def generate_visual_proof():
    print("\n" + "="*50)
    print(" 👁️ INITIATING VISION TRANSFORMER INTERPRETABILITY")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[System] Instantiating Multimodal Architecture...")
    model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
    
    # --- CRITICAL FIX: UNFREEZE THE BACKBONE ---
    # We must explicitly turn gradients back on to map the attention
    print("[System] Unfreezing ViT backbone for gradient calculation...")
    for param in model.visual_stream.parameters():
        param.requires_grad = True
        
    # model.load_state_dict(torch.load(r"D:\fyp\app\deepfake_detection\deepfake_detector_epoch_3.pth", map_location=device))
    model.load_state_dict(torch.load(r"D:\fyp\app\deepfake_detection\deepfake_detector_adapted_BEST.pth", map_location=device))
    
    cam = ViTGradCAM(model)
    
    # h5_path = r"D:\fyp\app\deepfake_detection\data\test.h5"
    h5_path = r"D:\fyp\app\deepfake_detection\data\dfdc_test.h5"
    print(f"[System] Extracting sample sequence from {h5_path}...")
    
    with h5py.File(h5_path, "r") as hf:
        visual = torch.tensor(hf["visual"]["0"][:]).unsqueeze(0).to(device)
        audio = torch.tensor(hf["audio"]["0"][:]).unsqueeze(0).to(device)
        
    # --- NEW: APPLY SPATIAL BLINDERS FOR INFERENCE ---
    # mask = torch.zeros((224, 224), device=device)
    # mask[32:192, 32:192] = 1.0
    # visual_masked = visual * mask.view(1, 1, 1, 224, 224)
    # -------------------------------------------------

    print("[System] Running forward/backward hooks to calculate ViT token gradients...")
    # Pass the masked visual to the heatmap generator
    heatmap = cam.generate_heatmap(visual, audio)
    
    # Detach visual tensor before converting to numpy to avoid memory trace issues
    first_frame = visual.squeeze()[0].detach().cpu().numpy()
    first_frame = np.transpose(first_frame, (1, 2, 0)) 
    
    first_frame = (first_frame - np.min(first_frame)) / (np.max(first_frame) - np.min(first_frame))
    
    heatmap_resized = cv2.resize(heatmap, (first_frame.shape[1], first_frame.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = (heatmap_colored / 255.0) * 0.4 + first_frame
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
    plt.title("ViT Attention: Audio-Visual Discrepancy Focus")
    plt.axis('off')
    plt.savefig("thesis_interpretability_heatmap.png", bbox_inches='tight')
    print("[System] ✅ Thesis artifact successfully generated: thesis_interpretability_heatmap.png")
    print("="*50)

if __name__ == "__main__":
    generate_visual_proof()