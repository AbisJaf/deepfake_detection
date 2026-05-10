import h5py
import matplotlib.pyplot as plt
import numpy as np
import random

def unnormalize_image(tensor):
    """
    Your preprocessing applied ImageNet normalization. 
    We must reverse this math so the image doesn't look dark and distorted to the human eye.
    """
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    # Reverse the normalization: Image = (Tensor * Std) + Mean
    img = tensor * std + mean
    
    # Clip values between 0 and 1 to prevent weird bright spots
    img = np.clip(img, 0, 1)
    
    # Matplotlib expects images in (Height, Width, Channels), but PyTorch uses (Channels, Height, Width)
    # We transpose the axes to fix this.
    return np.transpose(img, (1, 2, 0))

def visualize_random_sample():
    h5_path = r"D:\fyp\app\deepfake_detection\data\train.h5"
    
    with h5py.File(h5_path, 'r') as hf:
        total_videos = len(hf['labels'])
        
        # We need BOTH an integer (for the labels array) and a string (for the folder names)
        idx_int = random.randint(0, total_videos - 1)
        idx_str = str(idx_int)
        
        # Load the tensors (using the string for groups)
        vis_tensor = hf['visual'][idx_str][:]  # Shape: (16, 3, 224, 224)
        mouth_tensor = hf['mouth'][idx_str][:] # Shape: (16, 3, 224, 224)
        aud_tensor = hf['audio'][idx_str][:]   # Shape: (80, T)
        
        # Load the label (using the integer for the 1D array)
        label = hf['labels'][idx_int]
        
        label_text = "FAKE" if label == 1 else "REAL"
        
        # Set up a plotting grid
        fig, axes = plt.subplots(3, 5, figsize=(16, 9))
        fig.suptitle(f"Video Database Index: {idx_int} | Label: {label_text}", fontsize=18, fontweight='bold')
        
        # We have 16 frames, but let's just pick 5 evenly spaced frames to display
        frame_indices = np.linspace(0, 15, 5, dtype=int)
        
        for i, f_idx in enumerate(frame_indices):
            # 1. Plot Full Face
            face_img = unnormalize_image(vis_tensor[f_idx])
            axes[0, i].imshow(face_img)
            axes[0, i].axis('off')
            if i == 2: axes[0, i].set_title("Full Face Spatial Sequence", fontsize=14, pad=10)
            
            # 2. Plot Mouth Crop
            mouth_img = unnormalize_image(mouth_tensor[f_idx])
            axes[1, i].imshow(mouth_img)
            axes[1, i].axis('off')
            if i == 2: axes[1, i].set_title("Mouth Region Sequence", fontsize=14, pad=10)
            
        # 3. Plot Audio Spectrogram
        # Clear the bottom row of 5 individual plots to make one giant wide plot for the audio
        for ax in axes[2, :]:
            ax.remove()
        
        ax_audio = fig.add_subplot(3, 1, 3)
        # Use a thermal colormap ('magma') which is standard for audio spectrograms
        im = ax_audio.imshow(aud_tensor, aspect='auto', origin='lower', cmap='magma')
        ax_audio.set_title("80-Band Audio Mel-Spectrogram", fontsize=14)
        ax_audio.set_ylabel("Frequency Bands")
        ax_audio.set_xlabel("Time Frames")
        fig.colorbar(im, ax=ax_audio, format='%+2.0f dB')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_random_sample()