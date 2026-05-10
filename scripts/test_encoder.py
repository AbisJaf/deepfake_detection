import torch
import time
from models.visual_encoder import VisualEncoder
from models.audio_encoder import AudioEncoder

def run_encoder_diagnostics():
    print("\n" + "="*50)
    print(" 🧪 ENCODER ARCHITECTURE DIAGNOSTICS")
    print("="*50)
    
    batch_size = 2  # We test with a batch of 2 to ensure the models handle batches correctly
    target_frames = 16
    embed_dim = 512

    # 1. Initialize Models
    print("\n[Step 1] Initializing Encoders...")
    print("         (Note: ViT may take a moment to download weights if this is your first time)")
    try:
        vis_encoder = VisualEncoder(embed_dim=embed_dim)
        aud_encoder = AudioEncoder(embed_dim=embed_dim, target_frames=target_frames)
        print("✅ Models instantiated successfully.")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return

    # 2. Generate Dummy Data mimicking your HDF5 pipeline
    print("\n[Step 2] Generating simulated input tensors...")
    # Visual: (Batch, Frames, Channels, Height, Width)
    dummy_visual = torch.randn(batch_size, target_frames, 3, 224, 224)
    # Audio: (Batch, Mel_Bands, Time_Steps)
    dummy_audio = torch.randn(batch_size, 80, 391)
    
    print(f"   -> Visual Input: {dummy_visual.shape}")
    print(f"   -> Audio Input:  {dummy_audio.shape}")

    # 3. Execute Forward Pass
    print("\n[Step 3] Executing Forward Pass (No Gradients)...")
    try:
        start_time = time.time()
        with torch.no_grad(): # Disable gradients to save memory during testing
            vis_embedding = vis_encoder(dummy_visual)
            aud_embedding = aud_encoder(dummy_audio)
        end_time = time.time()
        print(f"✅ Forward pass completed in {round(end_time - start_time, 2)} seconds.")
    except Exception as e:
        print(f"❌ Forward pass crashed: {e}")
        return

    # 4. Verify Output Alignment
    print("\n[Step 4] Verifying Mathematical Alignment...")
    print(f"   -> Visual Output Shape: {vis_embedding.shape} | Target: ({batch_size}, 16, 512)")
    print(f"   -> Audio Output Shape:  {aud_embedding.shape} | Target: ({batch_size}, 16, 512)")
    
    if vis_embedding.shape == aud_embedding.shape == (batch_size, target_frames, embed_dim):
        print("\n🎉 ALL TESTS PASSED! Both streams are perfectly aligned and ready for fusion.")
    else:
        print("\n⚠️ WARNING: Shape mismatch detected. Cross-attention will fail.")

if __name__ == "__main__":
    run_encoder_diagnostics()