import io
import cv2
import random
import librosa
import numpy as np
from PIL import Image, ImageFilter

class CompressionAugmenter:
    def __init__(self):
        # Probabilities mapped directly from TDD Table 4.3
        self.p_jpeg = 0.50
        self.p_h264 = 0.40
        self.p_blur = 0.20
        self.p_drop = 0.15
        self.p_audio_noise = 0.20

    def apply_jpeg_compression(self, pil_image):
        """JPEG recompression: Quality factor q in [40, 80]"""
        if random.random() > self.p_jpeg:
            return pil_image
            
        q = random.randint(40, 80)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=q)
        img_byte_arr.seek(0)
        return Image.open(img_byte_arr)

    def apply_gaussian_blur(self, pil_image):
        """Gaussian blur: sigma in [0.5, 1.5]"""
        if random.random() > self.p_blur:
            return pil_image
            
        sigma = random.uniform(0.5, 1.5)
        return pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def apply_temporal_frame_dropping(self, frames):
        """Temporal frame-dropping: Drop p in {0.1, 0.2} frames"""
        if random.random() > self.p_drop:
            return frames
            
        drop_fraction = random.choice([0.1, 0.2])
        num_to_drop = int(len(frames) * drop_fraction)
        
        # Randomly select indices to drop
        drop_indices = random.sample(range(1, len(frames) - 1), num_to_drop)
        
        augmented_frames = []
        for i in range(len(frames)):
            if i in drop_indices:
                # Duplicate the previous frame to maintain T=16 sequence length
                augmented_frames.append(augmented_frames[-1])
            else:
                augmented_frames.append(frames[i])
                
        return augmented_frames

    def apply_audio_gaussian_noise(self, waveform):
        """Gaussian noise (audio): SNR in [15, 30] dB"""
        if random.random() > self.p_audio_noise:
            return waveform
            
        snr_db = random.uniform(15, 30)
        signal_power = np.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
        
        return waveform + noise

    def apply_visual_augmentations(self, frames):
        """Applies spatial visual augmentations (JPEG, Blur) to a list of frames."""
        # Frame dropping happens across the sequence
        frames = self.apply_temporal_frame_dropping(frames)
        
        # Spatial augs applied to each frame individually
        augmented_frames = []
        for frame in frames:
            frame = self.apply_jpeg_compression(frame)
            frame = self.apply_gaussian_blur(frame)
            augmented_frames.append(frame)
            
        return augmented_frames