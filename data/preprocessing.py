import cv2
import h5py
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from tqdm import tqdm
# from augemntaion import CompressionAugmenter
from data.augemntaion import CompressionAugmenter
import random
import subprocess
import tempfile
import os
import imageio_ffmpeg

class VisualPreprocessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.target_frames = 16
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)
        
        # ImageNet Statistics
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        self.full_face_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), self.normalize
        ])
        
        self.mouth_transform = transforms.Compose([
            transforms.Resize((96, 96)), 
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(), self.normalize
        ])

    def extract_uniform_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.target_frames:
            cap.release()
            return None
        indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def process_frames(self, frames):
        full_face_crops = []
        mouth_crops = []
        for frame in frames:
            boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
            if boxes is None or landmarks is None:
                continue 
            
            box = boxes[0].astype(int)
            face_crop = frame.crop((box[0], box[1], box[2], box[3]))
            
            left_mouth, right_mouth = landmarks[0][3], landmarks[0][4]
            mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            
            padding = 50 
            mouth_box = [
                int(mouth_center_x - padding), int(mouth_center_y - padding + 10), 
                int(mouth_center_x + padding), int(mouth_center_y + padding + 10)
            ]
            mouth_crop = frame.crop(tuple(mouth_box))
            
            full_face_crops.append(face_crop)
            mouth_crops.append(mouth_crop)
            
        if len(full_face_crops) == self.target_frames:
            return full_face_crops, mouth_crops
        return None, None


class AudioPreprocessor:
    def __init__(self):
        self.sr = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 80
        
        # Get the EXACT path to the built-in FFmpeg executable, bypassing Windows PATH
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def process_audio(self, audio_path, duration_sec=None):
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        try:
            # Use the absolute path to ffmpeg instead of just the word 'ffmpeg'
            command = [
                self.ffmpeg_exe, '-y', '-i', audio_path,
                '-vn',               # Drop the video stream
                '-acodec', 'pcm_s16le', # Standard 16-bit WAV encoding
                '-ar', str(self.sr), # 16000 Hz Sample Rate
                '-ac', '1',          # Mono audio
                temp_wav
            ]
            
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            y, _ = librosa.load(temp_wav, sr=self.sr, duration=duration_sec)
            return y
            
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except PermissionError:
                    pass
        
    def waveform_to_mel(self, waveform):
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=self.sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mean = np.mean(mel_db)
        std = np.std(mel_db)
        mel_norm = (mel_db - mean) / (std + 1e-6)
        
        return torch.tensor(mel_norm, dtype=torch.float32)

class DeepfakeDataProcessor:
    def __init__(self, split_csv, root_data_dir, output_h5_path, is_training=False):
        self.manifest = pd.read_csv(split_csv)
        self.root_dir = Path(root_data_dir)
        self.output_path = output_h5_path
        self.is_training = is_training
        
        self.vis_prep = VisualPreprocessor()
        self.aud_prep = AudioPreprocessor()
        self.augmenter = CompressionAugmenter()

    
    def run(self):
        print("\nIndexing all MP4 files in the dataset. This will take a few seconds...")
        path_map = {p.name: p for p in self.root_dir.rglob("*.mp4")}
        print(f"Successfully indexed {len(path_map)} video files.\n")

        with h5py.File(self.output_path, 'w') as hf:
            vis_group = hf.create_group('visual')
            mouth_group = hf.create_group('mouth')
            aud_group = hf.create_group('audio')
            labels_dset = hf.create_dataset('labels', (len(self.manifest),), maxshape=(None,), dtype='i')
            
            valid_idx = 0
            
            for index, row in tqdm(self.manifest.iterrows(), total=len(self.manifest)):
                filename = str(row['path']).split('/')[-1].split('\\')[-1] 
                
                if filename not in path_map:
                    continue
                    
                video_path_str = str(path_map[filename])
                label = int(row['binary_label'])
                
                # 1. Visual Stream Extraction
                raw_frames = self.vis_prep.extract_uniform_frames(video_path_str)
                if not raw_frames: 
                    continue
                    
                # 2. MTCNN Face Cropping
                face_crops, mouth_crops = self.vis_prep.process_frames(raw_frames)
                if not face_crops: 
                    continue
                
                # 3. Audio Extraction
                try:
                    waveform = self.aud_prep.process_audio(video_path_str)
                except Exception as e:
                    # If this prints, it means FFmpeg isn't installed properly
                    print(f"\n[AUDIO ERROR] Failed to process {filename}: {e}")
                    continue 
                
                # 4. Augmentation & Tensor Conversion
                if self.is_training and random.random() < 0.30:
                    face_crops = self.augmenter.apply_visual_augmentations(face_crops)
                    mouth_crops = self.augmenter.apply_visual_augmentations(mouth_crops)
                    waveform = self.augmenter.apply_audio_gaussian_noise(waveform)
                
                vis_tensors = torch.stack([self.vis_prep.full_face_transform(f) for f in face_crops])
                mouth_tensors = torch.stack([self.vis_prep.mouth_transform(m) for m in mouth_crops])
                aud_tensor = self.aud_prep.waveform_to_mel(waveform)
                
                # 5. Serialize
                vis_group.create_dataset(str(valid_idx), data=vis_tensors.numpy(), compression="lzf")
                mouth_group.create_dataset(str(valid_idx), data=mouth_tensors.numpy(), compression="lzf")
                aud_group.create_dataset(str(valid_idx), data=aud_tensor.numpy(), compression="lzf")
                labels_dset[valid_idx] = label
                
                valid_idx += 1
                
            labels_dset.resize((valid_idx,))

if __name__ == "__main__":
    # Example usage for Training Split
    ROOT_DIR = r"D:\fyp\dataset\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
    processor = DeepfakeDataProcessor(
        # split_csv=r"D:\fyp\app\deepfake_detection\data\splits\val_split.csv",
        split_csv=r"D:\fyp\dataset\4068245\DeepfakeTIMIT\DeepfakeTIMIT",
        root_data_dir=ROOT_DIR,
        output_h5_path=r"D:\fyp\app\deepfake_detection\data\timit_test.h5",
        is_training=False # Enables the 30% augmentation rule
    )
    processor.run()