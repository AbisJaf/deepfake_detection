import h5py
import numpy as np
from tqdm import tqdm

def create_balanced_benchmark():
    # 1. The Paths
    fakeavceleb_path = r"D:\fyp\app\deepfake_detection\data\test.h5"  # Has our Real videos
    timit_path = r"D:\fyp\app\deepfake_detection\data\timit_test.h5"  # Has our Fake videos
    output_path = r"D:\fyp\app\deepfake_detection\data\balanced_cross_test.h5"
    
    print("[System] Initiating Dataset Merger...")
    
    with h5py.File(output_path, 'w') as out_hf, \
         h5py.File(fakeavceleb_path, 'r') as fav_hf, \
         h5py.File(timit_path, 'r') as timit_hf:
        
        vis_grp = out_hf.create_group('visual')
        aud_grp = out_hf.create_group('audio')
        labels = []
        new_idx = 0
        
        # --- STEP 1: Extract 640 FAKES from DeepfakeTIMIT ---
        print("-> Pulling 640 Cross-Domain Fakes (DeepfakeTIMIT)...")
        timit_labels = timit_hf['labels'][:]
        for i in tqdm(range(len(timit_labels))):
            idx_str = str(i)
            # Copy data
            vis_grp.create_dataset(str(new_idx), data=timit_hf['visual'][idx_str][:], compression="gzip")
            aud_grp.create_dataset(str(new_idx), data=timit_hf['audio'][idx_str][:], compression="gzip")
            labels.append(1.0) # 1.0 = Fake
            new_idx += 1
            
        # --- STEP 2: Extract 640 REALS from FakeAVCeleb ---
        print("-> Pulling 640 Baseline Reals (FakeAVCeleb)...")
        fav_labels = fav_hf['labels'][:]
        real_indices = np.where(fav_labels == 0.0)[0] # Find where videos are Real
        
        # Take exactly 640 to match the fakes perfectly
        selected_reals = real_indices[:640] 
        
        for idx in tqdm(selected_reals):
            idx_str = str(idx)
            # Copy data
            vis_grp.create_dataset(str(new_idx), data=fav_hf['visual'][idx_str][:], compression="gzip")
            aud_grp.create_dataset(str(new_idx), data=fav_hf['audio'][idx_str][:], compression="gzip")
            labels.append(0.0) # 0.0 = Real
            new_idx += 1
            
        # Save the final label array
        out_hf.create_dataset('labels', data=np.array(labels, dtype=np.float32))
        
    print(f"\n✅ Success! Created perfectly balanced benchmark with {new_idx} total videos at: {output_path}")

if __name__ == "__main__":
    create_balanced_benchmark()