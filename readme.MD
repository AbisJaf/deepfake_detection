

### `README.md`


# VeriFace: Multimodal Deepfake Detection via Cross-Attention Audio-Visual Fusion

VeriFace is a multimodal Late-Fusion Dual-Encoder deepfake detection system developed to combat modern spatiotemporal generative models (e.g., Diffusion models, GANs). By jointly modelling audio and visual streams, VeriFace detects subtle cross-modal desynchronisations and phoneme-level lip-sync mismatches that survive heavy social media compression.

This repository contains the complete PyTorch implementation, custom data pipelines, and evaluation scripts for the Final Year Project completed at the Institute of Business & IT (IBIT), University of the Punjab.

## 🚀 Key Architectural Features
* **Dual-Stream Encoders:** Utilizes a Vision Transformer (ViT-B/16) for high-frequency spatial feature extraction and a custom 6-layer CNN for Mel-spectrogram audio encoding.
* **Cross-Attention Fusion:** A bidirectional transformer module that allows the visual and audio streams to attend to each other, localizing temporal desynchronization.
* **"Lights Out" Spatial Masking:** A data-centric intervention that blacks out the upper 55% of the face and peripheral edges, maintaining stable 224x224 ViT positional embeddings while eradicating the *Clever Hans* effect (background bias).
* **Hard Negative Mining (Temporal Shift):** Dynamically desynchronizes 50% of authentic videos during training (1/3 audio roll) to force the network to learn biometric lip-sync rather than visual-only artifacts.
* **Latent Space Anomaly Scoring:** Bypasses standard linear classification heads during cross-domain evaluation in favor of L2-normalized Euclidean distance scoring to prevent threshold squashing.

## 📊 Performance Metrics

| Evaluation Domain | Dataset | Accuracy | AUROC | Fakes Caught (Recall) |
| :--- | :--- | :--- | :--- | :--- |
| **In-Distribution** | FakeAVCeleb | 98.95% | 0.9785 | 85.5% |
| **Cross-Domain** | DFDC (OOD) | 62.25% | 0.6309 | 79.5% |

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/VeriFace-Deepfake-Detection.git](https://github.com/yourusername/VeriFace-Deepfake-Detection.git)
   cd VeriFace-Deepfake-Detection

```

2. Create a virtual environment and install the dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows use: .\env\Scripts\activate
pip install -r requirements.txt

```



## 📂 Repository Structure

```text
VeriFace/
├── data/                   # HDF5 dataset loaders and scripts
├── evaluate/               # Cross-domain evaluation protocols
│   └── evaluate.py         # Script for unseen data evaluation and EER calculation
├── models/                 # Architecture definitions
│   ├── detector.py         # Main MultimodalDeepfakeDetector wrapper
│   ├── visual_encoder.py   # ViT-B/16 implementation
│   ├── audio_encoder.py    # CNN-6 Mel-spectrogram encoder
│   └── cross_attention.py  # Bidirectional fusion module
├── scripts/                # Training and adaptation scripts
│   └── finetune_dfdc.py    # Domain adaptation fine-tuning script with continual learning monitors
├── training/               # Loss functions and optimizers
│   └── losses.py           # Composite BCE + L2-Normalized LSE-D loss
├── requirements.txt        # Python dependencies
└── README.md

```

## 🛠️ Usage

### Training / Domain Adaptation

To run the domain adaptation fine-tuning pipeline with the asymmetric curriculum and catastrophic forgetting monitor:

```bash
python scripts/finetune_dfdc.py

```

### Evaluation

To run the final exam evaluation on the unseen DFDC balanced test split using the latent space anomaly score:

```bash
python evaluate/evaluate.py

```
