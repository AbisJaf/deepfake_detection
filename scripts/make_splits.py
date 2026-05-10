import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def generate_splits_from_meta(csv_path: str, output_dir: str):
    print(f"Loading metadata from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Stratified Partitioning based on the COMBINATIONS
    # Assuming the column with the 4 combinations is named 'type'
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df['type'], # CRITICAL UPDATE: Stratifying on the 4 sub-classes
        random_state=42 
    )

    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['type'], # Maintaining the sub-class ratio again
        random_state=42
    )

    # 2. Binarize the labels AFTER the perfect split is guaranteed
    for dataset in [train_df, val_df, test_df]:
        dataset['binary_label'] = dataset['type'].apply(
            lambda x: 0 if x == 'RealVideo-RealAudio' else 1
        )

    # 3. Export Manifests
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Keep the combination 'type' in the CSV for debugging, plus the binary label
    columns_to_keep = ['path', 'type', 'binary_label']
    
    train_df[columns_to_keep].to_csv(out_path / "train_split.csv", index=False)
    val_df[columns_to_keep].to_csv(out_path / "val_split.csv", index=False)
    test_df[columns_to_keep].to_csv(out_path / "test_split.csv", index=False)

    print("\n--- Successful Sub-Class Balanced Split ---")
    print(f"Train Total: {len(train_df)}")
    print(train_df['type'].value_counts())
    
    print(f"\nVal Total:   {len(val_df)}")
    print(val_df['type'].value_counts())
    
    print(f"\nTest Total:  {len(test_df)}")
    print(test_df['type'].value_counts())

if __name__ == "__main__":
    # Using 'r' before the string treats backslashes as literal characters, perfect for Windows paths.
    META_CSV_PATH = r"D:\fyp\dataset\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\meta_data.csv" 
    
    # This will create a 'splits' folder inside your data directory to keep things organized
    OUTPUT_DIRECTORY = r"D:\fyp\app\deepfake_detection\data\splits"
    
    generate_splits_from_meta(META_CSV_PATH, OUTPUT_DIRECTORY)