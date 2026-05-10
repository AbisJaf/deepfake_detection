import h5py

def check_dataset_health():
    # Update this path if your file is located somewhere else
    # h5_path = r"D:\fyp\app\deepfake_detection\data\test.h5"
    h5_path = r"D:\fyp\app\deepfake_detection\data\test.h5"
    
    print("\n" + "="*40)
    print(" 📊 HDF5 Dataset Inspection")
    print("="*40)
    
    try:
        with h5py.File(h5_path, 'r') as hf:
            # The length of the labels array dictates the total valid videos
            total_success = len(hf['labels'])
            
            # print(f"Total Videos Attempted:   15,096")
            print(f"Total Videos Successful:  {total_success}")
            print(f"Failed/Skipped Videos:    {15096 - total_success}")
            print("-" * 40)
            
            # Print the mathematical shape of the very first video to verify integrity
            print(f"Visual Tensor Shape: {hf['visual']['0'].shape}")
            print(f"Mouth Tensor Shape:  {hf['mouth']['0'].shape}")
            print(f"Audio Tensor Shape:  {hf['audio']['0'].shape}")
            print(f"First Label:         {hf['labels'][0]}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_dataset_health()