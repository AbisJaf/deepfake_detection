import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.detector import MultimodalDeepfakeDetector


DATASET_NAME = "DFDC balanced test split"
CHECKPOINT_PATH = r"D:\fyp\app\deepfake_detection\deepfake_detector_adapted_BEST.pth"
H5_TEST_PATH = r"D:\fyp\app\deepfake_detection\data\test.h5"
# H5_TEST_PATH = r"D:\fyp\app\deepfake_detection\data\dfdc_test.h5"
INDEX_PATH = r"D:\fyp\app\deepfake_detection\data\dfdc_test_idx.npy"
BATCH_SIZE = 8
NUM_WORKERS = 4


class HDF5VideoDataset(Dataset):
    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as hf:
            total = len(hf["labels"])
        self.indices = np.asarray(indices if indices is not None else np.arange(total))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        with h5py.File(self.h5_path, "r") as hf:
            idx_str = str(real_idx)
            visual = torch.tensor(hf["visual"][idx_str][:])
            audio = torch.tensor(hf["audio"][idx_str][:])
            label = torch.tensor(hf["labels"][real_idx], dtype=torch.float32)
        return visual, audio, label


def pad_collate(batch):
    visuals, audios, labels = zip(*batch)
    visuals = torch.stack(visuals, 0)
    labels = torch.stack(labels, 0)
    max_len = max(a.shape[1] for a in audios)
    audios = torch.stack([F.pad(a, (0, max_len - a.shape[1])) for a in audios], 0)
    return visuals, audios, labels


def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[eer_idx], thresholds[eer_idx]


def best_youden_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    best_idx = np.argmax(tpr - fpr)
    return thresholds[best_idx]


def print_metrics(y_true, y_scores):
    y_pred = (y_scores >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    calibrated_threshold = best_youden_threshold(y_true, y_scores)
    calibrated_pred = (y_scores >= calibrated_threshold).astype(int)
    calibrated_acc = accuracy_score(y_true, calibrated_pred)
    calibrated_cm = confusion_matrix(y_true, calibrated_pred, labels=[0, 1])

    print(f"Accuracy @ 0.50 threshold:  {acc * 100:.2f}%")
    print(f"AUROC:                     {auroc:.4f}")
    print(f"EER:                       {eer * 100:.2f}% (threshold={eer_threshold:.4f})")
    print(f"Best threshold:            {calibrated_threshold:.4f}")
    print(f"Accuracy @ best threshold: {calibrated_acc * 100:.2f}%")
    print()
    print("Confusion matrix @ 0.50 threshold [[TN, FP], [FN, TP]]:")
    print(cm)
    print("Confusion matrix @ best threshold [[TN, FP], [FN, TP]]:")
    print(calibrated_cm)
    print()
    print(f"Real precision/recall/F1: {precision[0]:.4f} / {recall[0]:.4f} / {f1[0]:.4f}")
    print(f"Fake precision/recall/F1: {precision[1]:.4f} / {recall[1]:.4f} / {f1[1]:.4f}")


def evaluate_model():
    print("\n" + "=" * 50)
    print("INITIATING UNSEEN DATA EVALUATION PROTOCOL")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = np.load(INDEX_PATH) if INDEX_PATH else None

    print(f"[System] Dataset: {DATASET_NAME}")
    print(f"[System] HDF5: {H5_TEST_PATH}")
    print(f"[System] Checkpoint: {CHECKPOINT_PATH}")

    dataset = HDF5VideoDataset(H5_TEST_PATH, indices=indices)
    with h5py.File(H5_TEST_PATH, "r") as hf:
        # Load all labels into a numpy array first with [:], then index it
        all_labels = hf["labels"][:] 
        labels = all_labels[dataset.indices]
    print(
        f"[System] Test set: {len(dataset)} samples "
        f"({int((labels == 0).sum())} real, {int((labels == 1).sum())} fake)"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate,
    )

    print("[System] Instantiating architecture and loading weights...")
    model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_labels = []
    all_scores = []

    print("\n[System] Beginning inference...")
    with torch.no_grad():
        loop = tqdm(dataloader, leave=True)
        for visuals, audios, labels in loop:
            visuals = visuals.to(device)
            audios = audios.to(device)

            logits, _, _ = model(visuals, audios)
            probabilities = torch.sigmoid(logits)

            all_scores.extend(probabilities.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "=" * 50)
    print(f"FINAL EXAM RESULTS ({DATASET_NAME.upper()})")
    print("=" * 50)

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    print_metrics(y_true, y_scores)
    print("=" * 50)


if __name__ == "__main__":
    evaluate_model()
