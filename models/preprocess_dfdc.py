import os
import cv2
import librosa
import numpy as np
import h5py
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def build_dfdc_benchmark():
    # Update this path to wherever you extracted the Kaggle ZIP
    dfdc_dir = r"D:\fyp\dataset\archive\dfdc_train_part_46\dfdc_train_part_46"
    csv_path = r"D:\fyp\dataset\archive\filtered_metadata\filtered_metadata.csv" # Check the exact name of the CSV
    output_h5 = r"D:\fyp\app\deepfake_detection\data\dfdc_test.h5"
    
    print(f"[System] Loading DFDC Metadata from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[Error] Could not find the CSV! Check the filename.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with h5py.File(output_h5, 'w') as out_hf:
        vis_grp = out_hf.create_group('visual')
        aud_grp = out_hf.create_group('audio')
        labels = []
        valid_count = 0
        
        print("\n-> Initiating DFDC Unbiased Processing Pipeline...")
        loop = tqdm(df.iterrows(), total=len(df))
        
        for index, row in loop:
            try:
                v_filename = row['filename']
                # The label is usually a string in DFDC CSVs
                is_fake = 1.0 if str(row['label']).upper() == 'FAKE' else 0.0 
                
                v_path = os.path.join(dfdc_dir, v_filename)
                if not os.path.exists(v_path): continue
                
                # --- 1. Visual ---
                cap = cv2.VideoCapture(v_path)
                frames = []
                while len(frames) < 16:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame)
                    
                    boxes, _ = mtcnn.detect(pil_img)
                    if boxes is not None:
                        box = boxes[0].astype(int)
                        face = pil_img.crop((box[0], box[1], box[2], box[3]))
                        frames.append(transform(face))
                cap.release()
                
                if len(frames) < 16: continue
                v_tensor = torch.stack(frames)
                
                # --- 2. Audio ---
                y, sr = librosa.load(v_path, sr=16000)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # --- 3. Save ---
                idx_str = str(valid_count)
                vis_grp.create_dataset(idx_str, data=v_tensor.numpy(), compression="gzip")
                aud_grp.create_dataset(idx_str, data=mel_db, compression="gzip")
                labels.append(is_fake)
                
                valid_count += 1
                loop.set_postfix(Processed=valid_count)
                
            except Exception as e:
                continue
                
        out_hf.create_dataset('labels', data=np.array(labels, dtype=np.float32))
        print(f"\n✅ DFDC BENCHMARK CREATED! Total Valid Videos: {valid_count}")

if __name__ == '__main__':
    build_dfdc_benchmark()