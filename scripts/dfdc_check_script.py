# import h5py
# import numpy as np

# with h5py.File(r"D:\fyp\app\deepfake_detection\data\dfdc_test.h5", 'r') as hf:
#     labels = hf['labels'][:]

# real_idx = np.where(labels == 0)[0]  # 389 real
# fake_idx = np.where(labels == 1)[0]  # 1813 fake

# np.random.seed(42)

# # Reserve 200 real + 200 fake for testing
# test_real = np.random.choice(real_idx, size=200, replace=False)
# test_fake = np.random.choice(fake_idx, size=200, replace=False)
# test_idx  = np.concatenate([test_real, test_fake])
# np.random.shuffle(test_idx)

# # Remaining for adaptation
# adapt_real = np.setdiff1d(real_idx, test_real)  # 189 real
# adapt_fake = np.setdiff1d(fake_idx, test_fake)  # 1613 fake

# # Balance adaptation set too
# n = min(len(adapt_real), 189)
# adapt_fake_balanced = np.random.choice(adapt_fake, size=n, replace=False)
# adapt_idx = np.concatenate([adapt_real, adapt_fake_balanced])
# np.random.shuffle(adapt_idx)

# # Save both
# np.save(r"D:\fyp\app\deepfake_detection\data\dfdc_test_idx.npy",  test_idx)
# np.save(r"D:\fyp\app\deepfake_detection\data\dfdc_adapt_idx.npy", adapt_idx)

# print(f"Test set:       {len(test_idx)} samples (200 real + 200 fake)")
# print(f"Adaptation set: {len(adapt_idx)} samples ({n} real + {n} fake)")

import os
checkpoint_dir = r"D:\fyp\app\deepfake_detection"
ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
for c in ckpts:
    size = os.path.getsize(os.path.join(checkpoint_dir, c)) / (1024**3)
    print(f"{c} — {size:.2f} GB")