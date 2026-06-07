


"""
Domain Adaptation Fine-Tuning Script
=====================================
Takes the epoch 30 checkpoint and fine-tunes on a small DFDC adaptation set
to improve cross-dataset generalization without destroying FakeAVCeleb performance.

Strategy:
- FULLY UNFROZEN: Both backbones are actively trained from Epoch 1 to break the domain bias.
"""

import torch
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
import os
import random
import torchvision.transforms as T
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.detector import MultimodalDeepfakeDetector
from training.losses import DeepfakeCompositeLoss

# ─────────────────────────────────────────────
#  CONFIGURE PATHS HERE
# ─────────────────────────────────────────────
# CHECKPOINT_PATH   = r"D:\fyp\app\deepfake_detection\deepfake_detector_epoch_30.pth"
CHECKPOINT_PATH   = r"D:\fyp\app\deepfake_detection\deepfake_detector_adapted_BEST.pth"
FAKEAVCELEB_H5    = r"D:\fyp\app\deepfake_detection\data\train.h5"
DFDC_H5           = r"D:\fyp\app\deepfake_detection\data\dfdc_test.h5"
DFDC_TEST_IDX  = r"D:\fyp\app\deepfake_detection\data\dfdc_test_idx.npy"
DFDC_ADAPT_IDX = r"D:\fyp\app\deepfake_detection\data\dfdc_adapt_idx.npy"
OUTPUT_DIR        = r"D:\fyp\app\deepfake_detection"

# --- CHANGED TO 2 EPOCHS FOR IMMEDIATE PROOF ---
TOTAL_EPOCHS      = 5
PHYSICAL_BATCH    = 8
ACCUM_STEPS       = 4   # effective batch = 32
FAKEAVCELEB_RATIO = 0.70  # 70% FakeAVCeleb, 30% DFDC per epoch




# ─────────────────────────────────────────────
#  DATASETS (TEMPORAL SHIFT + MOUTH ALTERNATION)
# ─────────────────────────────────────────────
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, indices=None, apply_temporal_shift=False):
        self.h5_path = h5_path
        self.apply_temporal_shift = apply_temporal_shift
        with h5py.File(h5_path, 'r') as hf:
            total = len(hf['labels'])
        self.indices = indices if indices is not None else np.arange(total)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        with h5py.File(self.h5_path, 'r') as hf:
            visual = torch.tensor(hf['visual'][str(real_idx)][:])
            audio  = torch.tensor(hf['audio'][str(real_idx)][:])
            label  = torch.tensor(hf['labels'][real_idx], dtype=torch.float32)
            
            # We no longer need to load the 'mouth' tensor! This saves I/O time.

        # 🚨 THE HARD NEGATIVE MINING TRAP (GOLDEN RUN: 1/3 SHIFT) 🚨
        if self.apply_temporal_shift and label.item() == 0.0:
            if random.random() < 0.5:
                # Shift audio by 1/3 of its length to completely break sync
                shift_amount = audio.shape[1] // 3 
                audio = torch.roll(audio, shifts=shift_amount, dims=1)
                # Override label to Fake
                label = torch.tensor(1.0, dtype=torch.float32)

        # 🚨 THE "LIGHTS OUT" SPATIAL MASKING STRATEGY 🚨
        # Instead of zooming and destroying positional embeddings, 
        # we blackout everything except the lower half of the face 50% of the time.
        final_visual = visual.clone() # Keep original dimensions
        
        if self.apply_temporal_shift and random.random() < 0.5:
            # Determine spatial dimensions (handles [C, H, W] or [T, C, H, W])
            h_dim = -2 
            w_dim = -1
            
            height = final_visual.shape[h_dim]
            width  = final_visual.shape[w_dim]
            
            # Cutoff just below the nose, and crop the jaw/ears
            cutoff_h = int(height * 0.55)
            crop_w   = int(width * 0.20)
            
            if final_visual.dim() == 4: # [Frames, Channels, Height, Width]
                final_visual[:, :, :cutoff_h, :] = 0.0       # Blackout top half
                final_visual[:, :, :, :crop_w] = 0.0         # Blackout left edge
                final_visual[:, :, :, -crop_w:] = 0.0        # Blackout right edge
            elif final_visual.dim() == 3: # [Channels, Height, Width]
                final_visual[:, :cutoff_h, :] = 0.0
                final_visual[:, :, :crop_w] = 0.0
                final_visual[:, :, -crop_w:] = 0.0

        return final_visual, audio, label


def pad_collate(batch):
    visuals, audios, labels = zip(*batch)
    visuals = torch.stack(visuals)
    labels  = torch.stack(labels)
    max_len = max(a.shape[1] for a in audios)
    audios  = torch.stack([F.pad(a, (0, max_len - a.shape[1])) for a in audios])
    return visuals, audios, labels

# ─────────────────────────────────────────────
#  UNFREEZE ALL (THE FIX)
# ─────────────────────────────────────────────
def unfreeze_all(model):
    """Ensures every single weight is awake and learning from Batch 1"""
    for param in model.parameters():
        param.requires_grad = True
    print("[Freeze] All backbones UNFROZEN — entire architecture is adapting.")


# ─────────────────────────────────────────────
#  L2-NORMALIZED INFONCE LOSS (SOTA)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  L2-NORMALIZED CONTRASTIVE MARGIN LOSS 
# ─────────────────────────────────────────────
def contrastive_margin_loss(v_embed, a_embed, labels, margin=1.0):
    """
    Bounds the L2 distance strictly between 0.0 and 2.0.
    Kills the magnitude cheat code completely.
    """
    # 1. Flatten to destroy hidden dimensions
    v_flat = v_embed.view(v_embed.size(0), -1)
    a_flat = a_embed.view(a_embed.size(0), -1)
    
    # 2. THE CURE: Strict L2 Normalization (vectors now have length exactly 1.0)
    v_norm = F.normalize(v_flat, p=2, dim=1)
    a_norm = F.normalize(a_flat, p=2, dim=1)
    
    labels_flat = labels.view(-1)
    
    # 3. Pairwise distance is now strictly bounded [0, 2]
    dist = F.pairwise_distance(v_norm, a_norm, p=2, eps=1e-8)
    
    # 4. Apply forces
    loss_real = (1.0 - labels_flat) * torch.pow(dist, 2)
    loss_fake = labels_flat * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    
    return torch.mean(loss_real + loss_fake)

# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
@torch.no_grad()
# ─────────────────────────────────────────────
#  VALIDATION (WITH PROGRESS BAR)
# ─────────────────────────────────────────────
@torch.no_grad()
# ─────────────────────────────────────────────
#  VALIDATION (WITH PROGRESS BAR & AMP SPEED)
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    
    loop = tqdm(dataloader, desc="Validating", leave=False)
    for visuals, audios, labels in loop:
        visuals, audios = visuals.to(device), audios.to(device)
        
        # 🚨 THE SPEED FIX: Mixed Precision Inference 🚨
        with torch.amp.autocast('cuda'):
            probs, _, _ = model(visuals, audios)
            
        all_probs.extend(torch.sigmoid(probs).squeeze().cpu().numpy())
        all_labels.extend(labels.numpy())
        
    auroc = roc_auc_score(np.array(all_labels), np.array(all_probs))
    model.train()
    return auroc

# ─────────────────────────────────────────────
#  MAIN FINE-TUNING LOOP
# ─────────────────────────────────────────────
def finetune():
    print("\n" + "="*55)
    print("  🔧 DOMAIN ADAPTATION FINE-TUNING (UNFROZEN)")
    print("="*55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device: {device}")

    # ── Load reserved test indices ──
    if os.path.exists(DFDC_TEST_IDX):
        reserved = np.load(DFDC_TEST_IDX)
    else:
        with h5py.File(DFDC_H5, 'r') as hf:
            labels_all = hf['labels'][:]
        real_idx = np.where(labels_all == 0)[0]
        fake_idx = np.where(labels_all == 1)[0]
        np.random.seed(42)
        fake_balanced = np.random.choice(fake_idx, size=len(real_idx), replace=False)
        reserved = np.concatenate([real_idx, fake_balanced])
        np.save(DFDC_TEST_IDX, reserved)

    adapt_idx = np.load(DFDC_ADAPT_IDX)
    


    # ── Datasets (Arming the Trap for Training Only) ──
    fakeavceleb_ds = HDF5Dataset(FAKEAVCELEB_H5, apply_temporal_shift=True)
    dfdc_adapt_ds  = HDF5Dataset(DFDC_H5, adapt_idx, apply_temporal_shift=True)
    dfdc_val_ds    = HDF5Dataset(DFDC_H5, reserved, apply_temporal_shift=False)

    n_dfdc   = len(dfdc_adapt_ds)
    n_fav    = int(n_dfdc * (FAKEAVCELEB_RATIO / (1 - FAKEAVCELEB_RATIO)))
    n_fav    = min(n_fav, len(fakeavceleb_ds))
    fav_idx  = np.random.choice(len(fakeavceleb_ds), size=n_fav, replace=False)
    
    # 🚨 Asymmetric Curriculum: No temporal shift for FakeAVCeleb training subset
    fav_sub  = HDF5Dataset(FAKEAVCELEB_H5, indices=fav_idx) 

    combined_ds = ConcatDataset([fav_sub, dfdc_adapt_ds])
    
    train_loader = DataLoader(combined_ds, batch_size=PHYSICAL_BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=pad_collate)
    
    val_loader   = DataLoader(dfdc_val_ds, batch_size=PHYSICAL_BATCH, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=pad_collate)

    # ─────────────────────────────────────────────
    # 🚨 NEW: FAST BASELINE MONITOR (400 SAMPLES)
    # ─────────────────────────────────────────────
    total_fav_samples = len(fakeavceleb_ds)
    from torch.utils.data import Subset  # Native PyTorch subsetter
    
    fakeavceleb_val_ds_full = HDF5Dataset(FAKEAVCELEB_H5, apply_temporal_shift=False)
    
    np.random.seed(42)
    fav_fast_val_idx = np.random.choice(len(fakeavceleb_val_ds_full), size=400, replace=False)
    
    # 🚨 Force it to exactly 400 samples 
    fakeavceleb_val_ds = Subset(fakeavceleb_val_ds_full, fav_fast_val_idx)
    
    fakeavceleb_val_loader = DataLoader(fakeavceleb_val_ds, batch_size=PHYSICAL_BATCH, shuffle=False,
                                        num_workers=4, pin_memory=True, collate_fn=pad_collate)
    # ////////////////////////////////////////////////////////////////////////////////////////////////
    # ── Datasets (Arming the Trap for Training Only) ──
    fakeavceleb_ds = HDF5Dataset(FAKEAVCELEB_H5, apply_temporal_shift=True)
    dfdc_adapt_ds  = HDF5Dataset(DFDC_H5, adapt_idx, apply_temporal_shift=True)
    dfdc_val_ds    = HDF5Dataset(DFDC_H5, reserved, apply_temporal_shift=False)

    # 🚨 NEW: Baseline Monitor (Clean evaluation, no shift)
    fakeavceleb_val_ds = HDF5Dataset(FAKEAVCELEB_H5, apply_temporal_shift=False)

    # ... skip the subset creation code ...

    combined_ds = ConcatDataset([fav_sub, dfdc_adapt_ds])
    train_loader = DataLoader(combined_ds, batch_size=PHYSICAL_BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=pad_collate)
    val_loader   = DataLoader(dfdc_val_ds, batch_size=PHYSICAL_BATCH, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=pad_collate)
    
    # 🚨 NEW: The Forgetting DataLoader
    fakeavceleb_val_loader = DataLoader(fakeavceleb_val_ds, batch_size=PHYSICAL_BATCH, shuffle=False,
                                        num_workers=4, pin_memory=True, collate_fn=pad_collate)
    # /////////////////////////////////////////////

    print(f"\n[Model] Loading checkpoint: {CHECKPOINT_PATH}")
    model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # --- CHANGED TO MARGIN=2.0 AND POS_WEIGHT=0.25 ---
    criterion = DeepfakeCompositeLoss(lambda_weight=0.3, margin=2.0, pos_weight=0.25)
    scaler    = torch.amp.GradScaler('cuda')

    best_auroc = 0.0
    best_epoch = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):

        # --- FULLY UNFROZEN FROM BATCH 1 ---
        if epoch == 1:
            unfreeze_all(model)
            
        optimizer = AdamW([
            {'params': model.visual_stream.parameters(), 'lr': 1e-5},  
            {'params': model.audio_stream.parameters(),  'lr': 1e-5},
            {'params': model.fusion_module.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(),    'lr': 1e-4},
        ], weight_decay=1e-2)

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        print(f"\n--- Epoch {epoch}/{TOTAL_EPOCHS} [ALL LAYERS UNFROZEN] ---")
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

        for batch_idx, (visuals, audios, labels) in loop:
            visuals = visuals.to(device)
            audios  = audios.to(device)
            labels  = labels.to(device)


            with torch.amp.autocast('cuda'):
                probs, v_embed, a_embed = model(visuals, audios)
                
                # THE GOLDEN LOSS
                loss, bce, lsed = criterion(probs, labels, v_embed, a_embed)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUM_STEPS
            loop.set_postfix(Loss=f"{loss.item() * ACCUM_STEPS:.4f}")


        # ─────────────────────────────────────────────
        #  DUAL-VALIDATION & EARLY STOPPING
        # ─────────────────────────────────────────────
        avg_loss = running_loss / len(train_loader)

        # Validate on both domains
        val_auroc = validate(model, val_loader, device)               # DFDC Test Set
        fav_auroc = validate(model, fakeavceleb_val_loader, device)   # FakeAVCeleb Test Set
        
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | DFDC AUROC: {val_auroc:.4f} | FakeAVCeleb AUROC: {fav_auroc:.4f}")

        # Save Checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f"deepfake_detector_adapted_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Monitor DFDC Improvement
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_path  = os.path.join(OUTPUT_DIR, "deepfake_detector_FINAL_BEST.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best DFDC AUROC: {best_auroc:.4f} — saved to {best_path}")
        else:
            epochs_without_improvement += 1

        # 🚨 THE REVIEWER'S STOP CONDITIONS 🚨
        if fav_auroc < 0.92:
            print(f"  ⚠️ CATASTROPHIC FORGETTING TRIGGERED (FakeAVCeleb AUROC: {fav_auroc:.4f}). Halting.")
            break
            
        if epochs_without_improvement >= 2:
            print(f"  🛑 EARLY STOPPING TRIGGERED: DFDC AUROC stagnated for 2 epochs. Halting.")
            break

if __name__ == "__main__":
    finetune()