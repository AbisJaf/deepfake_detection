"""
Checkpoint Sweep Script
========================
Evaluates all saved checkpoints on both FakeAVCeleb and DFDC test sets.
Finds the optimal epoch that balances in-distribution and cross-dataset performance.

Run from project root:
    python scripts/checkpoint_sweep.py
"""

import torch
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.detector import MultimodalDeepfakeDetector

# ─────────────────────────────────────────────
#  CONFIGURE PATHS HERE
# ─────────────────────────────────────────────
CHECKPOINT_DIR     = r"D:\fyp\app\deepfake_detection"
FAKEAVCELEB_H5     = r"D:\fyp\app\deepfake_detection\data\test.h5"
DFDC_H5            = r"D:\fyp\app\deepfake_detection\data\dfdc_test.h5"
DFDC_TEST_IDX      = r"D:\fyp\app\deepfake_detection\data\dfdc_test_idx.npy"

# Which epochs to sweep (original training checkpoints)
EPOCHS_TO_SWEEP    = list(range(1, 31))  # epochs 1-30


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path
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
        return visual, audio, label


def pad_collate(batch):
    visuals, audios, labels = zip(*batch)
    visuals = torch.stack(visuals)
    labels  = torch.stack(labels)
    max_len = max(a.shape[1] for a in audios)
    audios  = torch.stack([F.pad(a, (0, max_len - a.shape[1])) for a in audios])
    return visuals, audios, labels


# ─────────────────────────────────────────────
#  EVALUATION FUNCTION
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []

    for visuals, audios, labels in dataloader:
        visuals, audios = visuals.to(device), audios.to(device)
        probs, _, _ = model(visuals, audios)
        all_probs.extend(torch.sigmoid(probs).squeeze().cpu().numpy())
        all_labels.extend(labels.numpy())

    y_true   = np.array(all_labels)
    y_scores = np.array(all_probs)
    y_pred   = (y_scores >= 0.5).astype(int)

    acc   = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    return auroc, acc, eer


# ─────────────────────────────────────────────
#  MAIN SWEEP
# ─────────────────────────────────────────────
def sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device: {device}")

    # ── Load datasets once ──
    print("\n[Data] Loading FakeAVCeleb test set...")
    fav_ds     = HDF5Dataset(FAKEAVCELEB_H5)
    fav_loader = DataLoader(fav_ds, batch_size=8, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=pad_collate)

    print("[Data] Loading DFDC test set...")
    dfdc_idx    = np.load(DFDC_TEST_IDX)
    dfdc_ds     = HDF5Dataset(DFDC_H5, dfdc_idx)
    dfdc_loader = DataLoader(dfdc_ds, batch_size=8, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=pad_collate)

    print(f"[Data] FakeAVCeleb: {len(fav_ds)} samples | DFDC: {len(dfdc_ds)} samples\n")

    # ── Results storage ──
    results = []

    print("=" * 75)
    print(f"{'Epoch':<8} {'FAV AUROC':<12} {'FAV Acc':<10} {'FAV EER':<10} {'DFDC AUROC':<12} {'AUROC Drop':<12}")
    print("=" * 75)

    best_combined = 0.0
    best_epoch    = 0

    for epoch in EPOCHS_TO_SWEEP:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"deepfake_detector_epoch_{epoch}.pth")

        if not os.path.exists(ckpt_path):
            print(f"Epoch {epoch:02d} | SKIPPED (checkpoint not found)")
            continue

        # Load model
        model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Evaluate on both datasets
        fav_auroc,  fav_acc,  fav_eer  = evaluate(model, fav_loader,  device)
        dfdc_auroc, dfdc_acc, dfdc_eer = evaluate(model, dfdc_loader, device)

        auroc_drop = fav_auroc - dfdc_auroc

        # Combined score: weighted average favouring FakeAVCeleb (70%) + DFDC (30%)
        combined = 0.7 * fav_auroc + 0.3 * dfdc_auroc

        results.append({
            'epoch':      epoch,
            'fav_auroc':  fav_auroc,
            'fav_acc':    fav_acc,
            'fav_eer':    fav_eer,
            'dfdc_auroc': dfdc_auroc,
            'drop':       auroc_drop,
            'combined':   combined,
        })

        marker = " ◄ BEST" if combined > best_combined else ""
        if combined > best_combined:
            best_combined = combined
            best_epoch    = epoch

        print(f"Ep {epoch:02d}   | {fav_auroc:.4f}      | {fav_acc*100:.1f}%    | {fav_eer*100:.1f}%    | {dfdc_auroc:.4f}     | -{auroc_drop:.4f}    {marker}")

        # Free VRAM between epochs
        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 75)
    print(f"  SWEEP COMPLETE")
    print(f"  Best balanced epoch: {best_epoch} (combined score: {best_combined:.4f})")
    print("=" * 75)

    best = next(r for r in results if r['epoch'] == best_epoch)
    print(f"\n  FakeAVCeleb AUROC: {best['fav_auroc']:.4f}")
    print(f"  DFDC AUROC:        {best['dfdc_auroc']:.4f}")
    print(f"  AUROC Drop:        -{best['drop']:.4f}")
    print(f"\n  Use: deepfake_detector_epoch_{best_epoch}.pth")

    # ── Also show top 5 by combined score ──
    print("\n  Top 5 epochs by combined score:")
    top5 = sorted(results, key=lambda x: x['combined'], reverse=True)[:5]
    for r in top5:
        print(f"  Epoch {r['epoch']:02d} | FAV: {r['fav_auroc']:.4f} | DFDC: {r['dfdc_auroc']:.4f} | Combined: {r['combined']:.4f}")


if __name__ == "__main__":
    sweep()