import os
import cv2
import librosa
import numpy as np
import h5py
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def build_timit_h5():
    # Your exact directory from the error log
    input_dir = r"D:\fyp\dataset\4068245\DeepfakeTIMIT\DeepfakeTIMIT"
    output_h5 = r"D:\fyp\app\deepfake_detection\data\timit_test.h5"
    
    print(f"[System] Scanning {input_dir} for videos...")
    videos = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.avi', '.mp4')):
                videos.append(os.path.join(root, file))
                
    if not videos:
        print("[Error] No videos found! Check your directory path.")
        return
        
    print(f"[System] Found {len(videos)} videos. Initializing Extractors...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, device=device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with h5py.File(output_h5, 'w') as hf:
        vis_grp = hf.create_group('visual')
        aud_grp = hf.create_group('audio')
        labels = []
        
        valid_count = 0
        loop = tqdm(videos, desc="Processing")
        
        for v_path in loop:
            try:
                # 1. Visual Extraction
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
                        # Crop face
                        face = pil_img.crop((box[0], box[1], box[2], box[3]))
                        frames.append(transform(face))
                cap.release()
                
                if len(frames) < 16: continue
                v_tensor = torch.stack(frames)
                
                # 2. Audio Extraction
                y, sr = librosa.load(v_path, sr=16000)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # 3. Save to H5
                idx_str = str(valid_count)
                vis_grp.create_dataset(idx_str, data=v_tensor.numpy(), compression="gzip")
                aud_grp.create_dataset(idx_str, data=mel_db, compression="gzip")
                
                # NOTE: DeepfakeTIMIT only contains fakes, so all labels are 1.0
                labels.append(1.0)
                valid_count += 1
                loop.set_postfix(Valid=valid_count)
                
            except Exception as e:
                continue
                
        hf.create_dataset('labels', data=np.array(labels, dtype=np.float32))
        print(f"\n[Success] Extracted {valid_count} videos into {output_h5}")

if __name__ == '__main__':
    build_timit_h5()