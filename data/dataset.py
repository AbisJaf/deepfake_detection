import h5py
import torch
import random
from torch.utils.data import Dataset

class DeepfakeHDF5Dataset(Dataset):
    def __init__(self, h5_file_path, apply_temporal_shift=False):
        """
        Args:
            h5_file_path (str): Path to the preprocessed .h5 file.
            apply_temporal_shift (bool): If True, randomly shifts audio of Real videos to create Hard Fakes.
        """
        self.h5_path = h5_file_path
        self.apply_temporal_shift = apply_temporal_shift
        
        # Read dataset length quickly without keeping file open
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = len(hf['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Fetches the pre-computed tensors for a specific index.
        """
        idx_str = str(idx)
        
        with h5py.File(self.h5_path, 'r') as hf:
            # Load numpy arrays and cast to PyTorch tensors
            visual_tensor = torch.from_numpy(hf['visual'][idx_str][:])
            mouth_tensor = torch.from_numpy(hf['mouth'][idx_str][:])
            audio_tensor = torch.from_numpy(hf['audio'][idx_str][:])
            label = torch.tensor(hf['labels'][idx], dtype=torch.float32)
            
        # =================================================================
        # 🚨 THE HARD NEGATIVE MINING TRAP (TEMPORAL SHIFT) 🚨
        # =================================================================
        # Only apply if the flag is enabled AND it is a REAL video (label == 0)
        if self.apply_temporal_shift and label.item() == 0.0:
            
            # Flip a coin: 50% chance to trigger the trap
            if random.random() < 0.5:
                # The audio shape is [80, T_frames]. We roll the time dimension (dim=1).
                # Shifting by 1/3 of the total frames guarantees it is completely out of sync.
                shift_amount = audio_tensor.shape[1] // 3 
                audio_tensor = torch.roll(audio_tensor, shifts=shift_amount, dims=1)
                
                # OVERRIDE THE LABEL: This perfectly lit, pristine video is now a FAKE.
                label = torch.tensor(1.0, dtype=torch.float32)
        # =================================================================
                
        return {
            'visual': visual_tensor, # Shape: [16, 3, 224, 224]
            'mouth': mouth_tensor,   # Shape: [16, 3, 224, 224]
            'audio': audio_tensor,   # Shape: [80, T_frames]
            'label': label           # Shape: [1] (0 for Real, 1 for Fake)
        }