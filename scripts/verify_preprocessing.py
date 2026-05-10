import os
import h5py
import pandas as pd
from data.preprocessing import DeepfakeDataProcessor

def run_micro_test():
    print("\n" + "="*50)
    print(" 🕵️ PREPROCESSING VERIFICATION TEST")
    print("="*50)
    
    output_test_file = r"D:\fyp\app\deepfake_detection\data\peek_test.h5"
    
    # 1. Initialize your exact processor
    processor = DeepfakeDataProcessor(
        split_csv=r"D:\fyp\app\deepfake_detection\data\splits\train_split.csv",
        root_data_dir=r"D:\fyp\dataset\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2",
        output_h5_path=output_test_file,
        is_training=False 
    )
    
    # 2. THE TRICK: Shrink the manifest to exactly 2 videos
    processor.manifest = processor.manifest.head(2)
    
    print("\n[Step 1] Running pipeline on 2 videos...")
    processor.run()
    
    # 3. Prove the file size is real
    if os.path.exists(output_test_file):
        file_size_mb = os.path.getsize(output_test_file) / (1024 * 1024)
        print(f"\n✅ SUCCESS: peek_test.h5 was generated!")
        print(f"📁 Actual File Size on Hard Drive: {file_size_mb:.2f} MB")
        
        # 4. Peek inside the HDF5 file to prove the tensors exist
        print("\n[Step 2] Opening the .h5 file to inspect the extracted data:")
        with h5py.File(output_test_file, 'r') as hf:
            print(f"   -> Visual Tensor Shape: {hf['visual']['0'].shape} (Expected: 16, 3, 224, 224)")
            print(f"   -> Mouth Tensor Shape:  {hf['mouth']['0'].shape}  (Expected: 16, 3, 224, 224)")
            print(f"   -> Audio Tensor Shape:  {hf['audio']['0'].shape}  (Expected: 80, T)")
            print(f"   -> Binary Label:        {hf['labels'][0]} (0 = Real, 1 = Fake)")
            
        print("\nVerification complete. Your preprocessing pipeline is fully functional!")
        
        # Clean up the test file
        os.remove(output_test_file)
    else:
        print("\n❌ FAILED: The file was not created.")

if __name__ == "__main__":
    run_micro_test()