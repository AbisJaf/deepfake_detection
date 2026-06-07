

import os
checkpoint_dir = r"D:\fyp\app\deepfake_detection"
ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
for c in ckpts:
    size = os.path.getsize(os.path.join(checkpoint_dir, c)) / (1024**3)
    print(f"{c} — {size:.2f} GB")