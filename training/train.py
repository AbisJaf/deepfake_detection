import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from scheduler import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import sys
import os

# Ensure Python can find the models folder from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.detector import MultimodalDeepfakeDetector
from training.losses import DeepfakeCompositeLoss

class HDF5VideoDataset(Dataset):
    """Custom DataLoader to stream directly from your HDF5 file to the GPU."""
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # Open temporarily just to get the total number of videos
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = len(hf['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # We open the file inside __getitem__ to prevent multi-threading crashes
        with h5py.File(self.h5_path, 'r') as hf:
            idx_str = str(idx)
            visual = torch.tensor(hf['visual'][idx_str][:])
            audio = torch.tensor(hf['audio'][idx_str][:])
            label = torch.tensor(hf['labels'][idx], dtype=torch.float32)
            return visual, audio, label


def pad_collate(batch):
    """
    Custom packing function for the DataLoader.
    Pads audio spectrograms with zeros so they all match the longest audio in the batch.
    """
    visuals, audios, labels = zip(*batch)
    
    # Visuals and labels are already uniform, so we can just stack them normally
    visuals = torch.stack(visuals, 0)
    labels = torch.stack(labels, 0)
    
    # Find the maximum time dimension (index 1) among the audio tensors in this batch
    max_len = max([a.shape[1] for a in audios])
    
    padded_audios = []
    for a in audios:
        # Calculate how many zeros we need to add to match the max_len
        pad_amount = max_len - a.shape[1]
        
        # F.pad pads from the back of the last dimension: (padding_left, padding_right)
        padded_a = F.pad(a, (0, pad_amount))
        padded_audios.append(padded_a)
        
    padded_audios = torch.stack(padded_audios, 0)
    
    return visuals, padded_audios, labels

def train_model():
    print("\n" + "="*50)
    print(" 🚀 INITIATING MULTIMODAL TRAINING SEQUENCE (RESUMING EPOCH 6)")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- NEW: GRADIENT ACCUMULATION SETTINGS ---
    physical_batch_size = 8    # What actually fits in your VRAM
    accumulation_steps = 4     # 8 * 4 = 32 (Your original effective batch size)
    total_epochs = 30
    h5_train_path = r"D:\fyp\app\deepfake_detection\data\train.h5"
    
    print(f"[System] Loading dataset from {h5_train_path}...")
    dataset = HDF5VideoDataset(h5_train_path)
    # Note: DataLoader now uses physical_batch_size (8)
    dataloader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=pad_collate)
    
    print("[System] Initializing Multimodal Architecture...")
    model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
    
    # --- NEW: LOAD EPOCH 5 CHECKPOINT ---
    print("[System] Loading saved weights from Epoch 5...")
    model.load_state_dict(torch.load("deepfake_detector_epoch_5.pth"))
    
    criterion = DeepfakeCompositeLoss(lambda_weight=0.3, margin=1.0)
    scaler = torch.amp.GradScaler('cuda') 

    optimizer = AdamW([
        {'params': model.visual_stream.parameters(), 'lr': 1e-4},
        {'params': model.audio_stream.parameters(), 'lr': 1e-4},
        {'params': model.fusion_module.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-2)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=5, total_epochs=total_epochs)
    
    # Fast-forward the scheduler to Epoch 6
    for _ in range(5):
        scheduler.step()

    # --- NEW: START LOOP AT EPOCH 6 ---
    for epoch in range(6, total_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} [PHASE 2: FULL FINE-TUNING (Unfrozen)] ---")
        
        # Unfreeze all layers for Phase 2
        for param in model.visual_stream.parameters(): param.requires_grad = True
        for param in model.audio_stream.parameters(): param.requires_grad = True

        model.train()
        running_loss, running_bce, running_lsed = 0.0, 0.0, 0.0
        
        # Clear the cache before the heavy lifting begins
        torch.cuda.empty_cache()
        optimizer.zero_grad() # Ensure gradients are totally clean

        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        for batch_idx, (visuals, audios, labels) in loop:
            visuals = visuals.to(device)
            audios = audios.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda'):
                probabilities, v_embed, a_embed = model(visuals, audios)
                loss, bce, lsed = criterion(probabilities, labels, v_embed, a_embed)
                
                # Normalize the loss because we are accumulating it over 4 steps
                loss = loss / accumulation_steps

            # Accumulate the gradients
            scaler.scale(loss).backward()

            # --- NEW: STEP OPTIMIZER ONLY EVERY 4 BATCHES ---
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Reset for the next 4 batches

            # Update metrics (multiply loss back up for accurate display)
            running_loss += (loss.item() * accumulation_steps)
            running_bce += bce.item()
            running_lsed += lsed.item()
            
            loop.set_postfix(Loss=(loss.item() * accumulation_steps), BCE=bce.item(), LSED=lsed.item())

        print(f"Epoch {epoch} Complete | Avg Loss: {running_loss/len(dataloader):.4f}")
        
        scheduler.step()
        torch.save(model.state_dict(), f"deepfake_detector_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_model()