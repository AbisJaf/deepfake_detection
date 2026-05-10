import h5py
import torch
from torch.utils.data import Dataset

class DeepfakeHDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        """
        Args:
            h5_file_path (str): Path to the preprocessed .h5 file.
        """
        self.h5_path = h5_file_path
        
        # Read dataset length quickly without keeping file open
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = len(hf['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Fetches the pre-computed tensors for a specific index.
        File is opened inside __getitem__ to support PyTorch DataLoader multiprocessing.
        """
        idx_str = str(idx)
        
        with h5py.File(self.h5_path, 'r') as hf:
            # Load numpy arrays and cast to PyTorch tensors
            visual_tensor = torch.from_numpy(hf['visual'][idx_str][:])
            mouth_tensor = torch.from_numpy(hf['mouth'][idx_str][:])
            audio_tensor = torch.from_numpy(hf['audio'][idx_str][:])
            
            # Fetch binary label
            label = torch.tensor(hf['labels'][idx], dtype=torch.float32)
            
        return {
            'visual': visual_tensor, # Shape: [16, 3, 224, 224]
            'mouth': mouth_tensor,   # Shape: [16, 3, 224, 224]
            'audio': audio_tensor,   # Shape: [80, T_frames]
            'label': label           # Shape: [1] (0 for Real, 1 for Fake)
        }