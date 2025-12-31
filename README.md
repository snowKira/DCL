
# Improving Color Image Steganalysis with Dual Contrastive Learning (DCL)

This repository contains the official implementation of the paper **"Improving Color Image Steganalysis with Dual Contrastive Learning"**.

The **Dual Contrastive Learning (DCL)** framework is a general method designed to enhance the performance and training stability of color image steganalysis models. By integrating features from both the preprocessing (shallow) and representation (deep) modules, DCL captures both high-frequency details and abstract semantic information.

---

## ✨ Key Features

- **Dual-Stage Contrastive Learning**: A two-stage training process that independently leverages shallow features (for fine-grained noise) and deep features (for abstract representations).
- **Novel Contrastive Loss**: A loss function that simultaneously optimizes intra-class compactness and inter-class separation.
- **Adaptive Weight Strategy**: A dynamic balancing mechanism using a Softmax-based approach to harmonize classification loss and contrastive loss during training.
- **Model Agnostic**: A universal technique applicable to any CNN-based steganalysis framework (e.g., SR-NET, UC-NET, WISER-NET).
- **Superior Performance**: Significant improvements in detection accuracy and convergence speed across both JPEG and Spatial domains.

---

## 🏗️ Methodology Overview

### 1. Dual Contrastive Framework
The framework divides training into two phases:
*   **Stage 1 (Blue Path)**: Focuses on features from the Preprocessing module to capture low-level steganographic signals.
*   **Stage 2 (Green Path)**: Focuses on features from the Feature Representation module to capture high-level discriminative patterns.

### 2. Adaptive Weight Strategy
To prevent suboptimal convergence, we use an adaptive weight $\lambda$ calculated from the loss values of previous iterations:
$$L_t = K^{mod} \cdot L^{mod}_t + K^{steg} \cdot L^{steg}_t$$

---

## 📊 Experimental Results

Experiments conducted on **ALASKA II** and **BOSSBase** datasets demonstrate:
*   **Accuracy Boost**: Up to **2.31%** improvement for UERD (JPEG domain) on SR-NET.
*   **Stability**: Faster convergence and smaller loss fluctuations compared to standard training.
*   **Efficiency**: Nearly negligible computational overhead (only ~1-2 MiB additional memory).

| Model | Domain | Metric | Baseline | **With DCL** |
| :--- | :--- | :--- | :--- | :--- |
| SR-NET | JPEG (0.4bpp) | Acc | 82.81% | **84.79%** |
| UC-NET | Spatial (0.4bpp) | Acc | 78.27% | **78.62%** |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU (2080TI or better recommended)

---


## 🤝 Acknowledgements
This work builds upon several advanced steganalysis baselines including SR-NET, UC-NET, and WISER-NET. We thank the authors of these works for their contributions to the community.
