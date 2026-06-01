import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from evaluate import HDF5VideoDataset, pad_collate
from models.detector import MultimodalDeepfakeDetector
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_ablation(mode="visual_only"):
    print("\n" + "="*50)
    print(f" 🔬 INITIATING ABLATION STUDY: {mode.upper()}")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h5_test_path = r"D:\fyp\app\deepfake_detection\data\dfdc_test.h5" 
    checkpoint_path = r"D:\fyp\app\deepfake_detection\deepfake_detector_epoch_30.pth"

    dataset = HDF5VideoDataset(h5_test_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, collate_fn=pad_collate)

    model = MultimodalDeepfakeDetector(embed_dim=512).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_labels, all_probs = [], []

    with torch.no_grad():
        loop = tqdm(dataloader, leave=True)
        for visuals, audios, labels in loop:
            visuals = visuals.to(device)
            audios = audios.to(device)

            # --- ABLATION LOGIC ---
            if mode == "visual_only":
                # Zero out the audio tensor completely
                audios = torch.zeros_like(audios).to(device)
            elif mode == "audio_only":
                # Zero out the visual tensor completely
                visuals = torch.zeros_like(visuals).to(device)

            logits, _, _ = model(visuals, audios)
            probabilities = torch.sigmoid(logits)
            
            all_probs.extend(probabilities.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    try:
        auroc = roc_auc_score(all_labels, all_probs)
        y_pred = (torch.tensor(all_probs) >= 0.5).int().numpy()
        acc = accuracy_score(all_labels, y_pred)
        
        print(f"\n🎯 {mode.upper()} Accuracy: {acc * 100:.2f}%")
        print(f"🔬 {mode.upper()} AUROC:    {auroc:.4f}")
    except ValueError:
        print("\n[Error] AUROC cannot be calculated (only one class present).")

if __name__ == "__main__":
    # Run the visual-only test
    evaluate_ablation(mode="visual_only")
