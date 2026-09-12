# SA-UNetv2: Comprehensive Research & Reproduction Reference Guide

> **Paper Title:** *SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation*  
> **Conference:** Proceedings of the IEEE International Symposium on Biomedical Imaging (**ISBI 2026, Oral Presentation**)  
> **Authors:** Changlu Guo$^1$, Anders Nymark Christensen$^1$, Anders Bjorholm Dahl$^1$, Yugen Yi$^2$, Morten Rieger Hannemose$^1$  
> **Affiliations:** $^1$Department of Applied Mathematics and Computer Science, Technical University of Denmark (DTU); $^2$School of Artificial Intelligence, Jiangxi Normal University  
> **Links:** [arXiv:2509.11774](https://arxiv.org/abs/2509.11774) | [Official Project Page](https://clguo.github.io/SA-UNetv2) | [GitHub Code](https://github.com/clguo/SA-UNetv2)  
> **Reference Companion Paper:** *ACE-ProtoNet: Adaptive covariance eigen-gate and uncertainty-aware prototype learning for coronary artery segmentation* (Medical Image Analysis / MIA 2026)

---

## 1. Executive Summary & Core Motivation

### 1.1 Clinical Context
Retinal vasculature is a critical biomarker for early, non-invasive diagnosis of systemic and ocular conditions such as:
- Diabetic retinopathy (microaneurysms, neovascularization)
- Hypertension (arteriolar narrowing, nicking)
- Arteriosclerosis and cardiovascular risk
- Neurodegenerative disorders (e.g., Alzheimer's disease)

### 1.2 Key Challenges in Retinal Vessel Segmentation
1. **Severe Class Imbalance:** Vessel pixels comprise **less than 10%** of total pixels (extreme foreground-background skew). Standard losses like Binary Cross-Entropy (BCE) suffer from heavy background bias and miss faint vessels.
2. **Fine Geometric Complexity:** Vessels form intricate, multi-scale tree-like branching networks with tiny capillaries and low-contrast regions resulting from non-uniform fundus illumination and pathologies.
3. **Clinical Deployment Constraints:** Existing high-accuracy models (AG-Net, IterNet, MamUNet) are excessively large (**8M–25M parameters**), requiring expensive GPUs. Conversely, ultra-lightweight models (e.g., RetinalLiteNet, 0.066M) compromise segmentation fidelity significantly.

### 1.3 Key Innovations of SA-UNetv2 Over SA-UNet (v1)
| Dimension | Original SA-UNet (ICPR 2020) | SA-UNetv2 (ISBI 2026) | Rationale |
| :--- | :--- | :--- | :--- |
| **Skip Attention** | Attention only in the bottleneck; skips are plain concatenations | **Cross-scale Spatial Attention (CSA)** on all skip connections | Fuses encoder details with decoder context, bridging semantic gap |
| **Conv Block** | `Conv3x3` $\to$ `DropBlock` $\to$ `BN` $\to$ `ReLU` | `Conv3x3` $\to$ `DropBlock` $\to$ `GroupNorm` $\to$ `SiLU` | Batch-size independent stability (GN); smooth gradient flow for low-contrast edges (SiLU) |
| **Channels** | $[16, 32, 64, 128]$ | **$[16, 32, 48, 64]$** | **51.9% parameter reduction** ($0.54\text{M} \to 0.26\text{M}$); eliminates channel redundancy |
| **Loss Function** | Binary Cross-Entropy (BCE) alone | **Compound Loss:** $0.5 \times \text{BCE} + 0.5 \times \text{MCC}$ | Differentiable continuous MCC balances extreme foreground/background imbalance |
| **Efficiency** | $0.54\text{M}$ params, $26.54$ GFLOPs, $1.12\text{s}$ CPU | **$0.26\text{M}$ params, $21.19$ GFLOPs, $0.95\text{s}$ CPU** | Sub-second real-time inference on low-cost CPUs with higher accuracy |

---

## 2. Complete Architecture Specification

### 2.1 Macro Topology
SA-UNetv2 uses a 4-level encoder-decoder U-Net backbone with:
- Channel progression: **16 $\to$ 32 $\to$ 48 $\to$ 64** (Bottleneck = 64).
- Standard downsampling: $2\times 2$ Max Pooling.
- Standard upsampling: $2\times 2$ Transposed Convolution (`Conv2DTranspose`, kernel $3\times 3$, stride 2, padding `same`).
- Bottleneck: Standard **Spatial Attention (SA)** module.
- Skip Connections: **Cross-scale Spatial Attention (CSA)** modules at all 3 skip levels ($32, 48, 64$ channels).

```
Input (592 x 592 x 3)
   │
   ▼
[Level 1: 16 ch] ── ConvBlock x2 ──────────────────────────┐
   │                                                       │
   ▼ (MaxPool 2x2)                                         │
[Level 2: 32 ch] ── ConvBlock x2 ─────────────┐            │
   │                                          │            │
   ▼ (MaxPool 2x2)                            │            │
[Level 3: 48 ch] ── ConvBlock x2 ┐            │            │
   │                             │            │            │
   ▼ (MaxPool 2x2)               ▼ (CSA)      ▼ (CSA)      ▼ (CSA)
[Level 4: 64 ch] ── ConvBlock ───┴────────────┴────────────┴──────
   │                 + SA Module
   ▼
Bottleneck (64 ch)
   │
   ▼ (ConvTranspose2D 2x2)
[Decoder 3: 48 ch] ── Concat[Up, CSA(Enc3, Up)] ── ConvBlock x2
   │
   ▼ (ConvTranspose2D 2x2)
[Decoder 2: 32 ch] ── Concat[Up, CSA(Enc2, Up)] ── ConvBlock x2
   │
   ▼ (ConvTranspose2D 2x2)
[Decoder 1: 16 ch] ── Concat[Up, CSA(Enc1, Up)] ── ConvBlock x2
   │
   ▼
Conv2D 1x1 (1 filter, linear) ── Sigmoid ── Output (592 x 592 x 1)
```

---

### 2.2 Micro Block Design

#### 1. Optimized Convolutional Block
Every encoder and decoder stage contains **two consecutive Conv Blocks**, each composed of:
$$\text{Input} \xrightarrow{\text{Conv } 3\times 3} \xrightarrow{\text{DropBlock2D}} \xrightarrow{\text{GroupNormalization}} \xrightarrow{\text{SiLU}} \text{Output}$$

- **`Conv2D`**: Kernel size $(3, 3)$, padding `'same'`, initialization `'he_normal'`, no bias when followed by normalization.
- **`DropBlock2D`**: Regularization applied directly onto feature maps to drop structured contiguous regions.
  - `block_size`: **7**
  - `rate` (dropout probability): **0.15**
- **`GroupNormalization`**: 
  - `groups`: **8** (works stably across channel sizes 16, 32, 48, 64 because each is divisible by 8: channels per group are 2, 4, 6, 8 respectively).
  - Eliminates batch-size dependency during training, essential for small batch sizes ($N=8$ or $N=2$).
- **`SiLU` Activation**: $\text{SiLU}(x) = x \cdot \sigma(x)$. Smooth, non-monotonic gradient flow prevents dead neurons and captures subtle contrast gradients along vessel edges.

---

### 2.3 Attention Modules

#### 1. Bottleneck Spatial Attention (SA Module)
Applied in the bottleneck layer between the two conv blocks:
$$F_{SA} = F \cdot \sigma\left( f^{7\times 7}\left( \text{AvgPool}_c(F) \right) \right)$$
- Input: Feature map $F \in \mathbb{R}^{H \times W \times C}$
- Channel average pooling along axis 3: $\text{AvgPool}_c(F) \in \mathbb{R}^{H \times W \times 1}$
- Convolution: $7\times 7$ Conv2D (1 filter, padding `'same'`, no bias)
- Activation: Sigmoid $\sigma(\cdot)$
- Multiply: Element-wise scaling of original $F$ by the spatial attention mask.
- Parameters: $7 \times 7 \times 1 \times 1 = 49$ parameters (or 98 if 2 channels).

#### 2. Cross-scale Spatial Attention (CSA Module)
Placed on skip connections between encoder feature $F_e$ and upsampled decoder feature $F_d$:
$$F_{out} = F_e \cdot \sigma\Big( f^{7\times 7}\big( [\,\text{AvgPool}_c(F_e) \,;\, \text{AvgPool}_c(F_d)\,] \big)\Big)$$

- **Inputs**: 
  - $F_e \in \mathbb{R}^{H \times W \times C_{enc}}$ (high-resolution spatial details from encoder)
  - $F_d \in \mathbb{R}^{H \times W \times C_{dec}}$ (deep contextual guidance from upsampled decoder)
- **Channel Average Pooling**:
  - $A_e = \text{mean}(F_e, \text{axis}=3, \text{keepdims}=\text{True}) \in \mathbb{R}^{H \times W \times 1}$
  - $A_d = \text{mean}(F_d, \text{axis}=3, \text{keepdims}=\text{True}) \in \mathbb{R}^{H \times W \times 1}$
- **Concatenation**: $A_{cat} = [A_e, A_d] \in \mathbb{R}^{H \times W \times 2}$
- **Spatial Filtering**: $\text{AttnMap} = \sigma\big(\text{Conv2D}_{7\times 7, \text{filters}=1, \text{padding='same', no\_bias}}(A_{cat})\big) \in \mathbb{R}^{H \times W \times 1}$
- **Gated Feature**: $F_{out} = F_e \odot \text{AttnMap}$
- **Parameter Cost**: Exactly **98 parameters** ($7 \times 7 \times 2 \times 1 = 98$).
- **Decoder Concatenation**: The decoder then concatenates $[F_d, F_{out}]$ as input to its next conv block.

---

## 3. Mathematical Loss Formulation

Retinal vessel pixels comprise $< 10\%$ of total image area. SA-UNetv2 addresses this by uniting local pixel accuracy with global correlation.

### 3.1 Binary Cross-Entropy Loss ($L_{BCE}$)
Optimizes local, pixel-wise classification fidelity:
$$L_{BCE} = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \Big]$$
Where:
- $y_i \in \{0, 1\}$ is the binary ground-truth label for pixel $i$.
- $p_i \in [0, 1]$ is the continuous predicted probability for pixel $i$.
- $N$ is total pixels in the batch/image.

### 3.2 Differentiable Matthews Correlation Coefficient Loss ($L_{MCC}$)
MCC measures the quality of binary classifications even when classes are heavily unbalanced. SA-UNetv2 formulates a **continuous, soft, differentiable surrogate** without thresholding:

1. **Continuous Confusion Matrix Quantities**:
   $$TP = \sum_{i=1}^N p_i \cdot y_i$$
   $$TN = \sum_{i=1}^N (1 - y_i) \cdot (1 - p_i)$$
   $$FP = \sum_{i=1}^N (1 - y_i) \cdot p_i$$
   $$FN = \sum_{i=1}^N y_i \cdot (1 - p_i)$$

2. **Differentiable MCC Objective**:
   $$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)} + \epsilon}$$
   where $\epsilon = 10^{-7}$ prevents numerical instability / division by zero.

3. **MCC Loss Definition**:
   $$L_{MCC} = 1 - MCC$$
   *(Since $MCC \in [-1, 1]$, $L_{MCC} \in [0, 2]$. Minimizing $L_{MCC}$ maximizes correlation with the minority class).*

### 3.3 Compound Objective
$$L_{total} = \lambda_1 L_{BCE} + \lambda_2 L_{MCC}$$
- Optimal paper setting: **$\lambda_1 = 0.5, \lambda_2 = 0.5$** (equal weighting).

---

## 4. Dataset Protocols & Preprocessing Pipeline

### 4.1 Datasets
| Dataset | Total Images | Train / Test Split | Native Resolution | Padded Resolution | FOV Mask Available? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DRIVE** | 40 | 20 Train / 20 Test | $584 \times 565 \times 3$ | **$592 \times 592 \times 3$** | Yes (official circular mask) |
| **STARE** | 20 | 16 Train / 4 Test | $700 \times 605 \times 3$ | **$704 \times 704 \times 3$** | No official mask |

### 4.2 Padding and Cropping Protocol
- Networks with 3 max-pooling operations require spatial dimensions divisible by $2^3 = 8$.
- **DRIVE:** $584 \times 565 \xrightarrow{\text{zero pad}} 592 \times 592$
  $$\text{padded\_img}[:584, :565, :] = \text{orig\_img}$$
- **STARE:** $700 \times 605 \xrightarrow{\text{zero pad}} 704 \times 704$
- **Testing Crop:** After model prediction, probabilities are cropped back to original shape ($584 \times 565$ or $700 \times 605$) prior to evaluation.

### 4.3 Data Augmentation & Validation Split
- Follows the official SA-UNet augmentation protocol:
  - Random rotations, horizontal/vertical flips, elastic deformations, random scaling.
- **Validation Split:** 10% of augmented training samples are held out for validation during training.
- Pixel normalization: $X \in [0, 1]$ via `img.astype('float32') / 255.0`.
- Target binarization: $Y \in \{0.0, 1.0\}$ via threshold at 127 (`cv2.threshold(label, 127, 255, THRESH_BINARY) / 255.0`).

---

## 5. Training & Optimization Specifications

| Parameter | Exact Value / Strategy |
| :--- | :--- |
| **Optimizer** | Adam |
| **Initial Learning Rate** | $1 \times 10^{-3}$ ($0.001$) |
| **Loss Function** | $0.5 \times L_{BCE} + 0.5 \times L_{MCC}$ |
| **Batch Size** | **8** for DRIVE, **2** for STARE |
| **Max Epochs** | **150** |
| **Learning Rate Scheduler** | `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)` |
| **Early Stopping** | `EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)` |
| **Best Model Checkpoint** | `ModelCheckpoint(monitor='val_accuracy' or 'val_loss', save_best_only=True)` |
| **DropBlock Params** | `block_size = 7`, `rate = 0.15` |
| **GroupNorm Params** | `groups = 8` |
| **Binarization Threshold** | $\tau = 0.5$ for inference metrics |

---

## 6. Evaluation Protocols & Formulas

### 6.1 Mathematical Definitions
From the binary confusion matrix: $TP, TN, FP, FN$:
1. **F1-Score / Dice Coefficient**:
   $$F1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
2. **Jaccard Index / IoU**:
   $$\text{Jaccard} = \frac{TP}{TP + FP + FN} = \frac{F1}{2 - F1}$$
3. **Sensitivity / Recall / True Positive Rate (TPR)**:
   $$\text{Sen} = \frac{TP}{TP + FN}$$
4. **Specificity / True Negative Rate (TNR)**:
   $$\text{Spe} = \frac{TN}{TN + FP}$$
5. **Accuracy**:
   $$\text{ACC} = \frac{TP + TN}{TP + TN + FP + FN}$$
6. **Matthews Correlation Coefficient (MCC)**:
   $$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
7. **Area Under ROC Curve (AUC-ROC)**:
   Computed using continuous predicted probabilities ($p_i \in [0, 1]$) against ground-truth labels.

### 6.2 Dual Evaluation Protocols on DRIVE
1. **Without FOV Mask (Primary Benchmark Protocol)**:
   - Metrics are computed over the full rectangular image area ($584 \times 565 = 329,960$ pixels per image).
   - This protocol is strictly standard for clinical screening pipelines where black border regions are present.
2. **With FOV Mask**:
   - Metrics are computed **only** on pixels lying strictly inside the circular Field-of-View mask (`mask == 1`).
   - All black background outside the retina circle is excluded from the confusion matrix.

---

## 7. Official Benchmark Results (Full Numerical Tables)

### Table 1: Comprehensive Comparison on DRIVE Dataset
*(† indicates results cited from MamUNet [ISBI 2025])*

#### A. Without FOV Mask
| Model | Venue | F1 (%) | Jacc (%) | Sen (%) | Spe (%) | ACC (%) | MCC (%) | AUC (%) | GFLOPs | Params (M) | Mem (MB) | CPU Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| U-Net | MICCAI'15 | 81.30 | 68.52 | 80.65 | 98.34 | 96.77 | 79.66 | 98.24 | 137.10 | 8.64 | 34.72 | 3.16 |
| Attention U-Net | MIDL'18 | 81.47 | 68.76 | 81.24 | 98.30 | 96.78 | 79.84 | 98.22 | 425.05 | 8.65 | 34.80 | 7.74 |
| AG-Net | MICCAI'19 | — | 69.65 | 81.00 | 98.48 | 96.92 | 79.84 | 98.56 | — | 9.34 | 37.36 | — |
| U-Net++ | TMI'20 | 81.18 | 68.34 | 82.40 | 98.07 | 96.67 | 79.50 | 98.44 | 338.25 | 10.20 | 41.08 | 6.95 |
| PA-Filter | ISBI'22 | 82.61 | — | — | — | 96.99 | — | 98.43 | — | 2.01 | — | — |
| UNet 3+ | ICASSP'20 | 81.46 | 68.74 | 82.02 | 98.18 | 96.75 | 79.79 | 98.47 | 198.20 | 1.97 | 8.36 | 5.29 |
| ACC-UNet-Lite | MICCAI'23 | 81.26 | 68.47 | 82.05 | 98.14 | 96.71 | 79.57 | 98.34 | 215.59 | 9.14 | 37.05 | 4.41 |
| nnWNet | CVPR'25 | 82.18 | 69.86 | — | — | — | — | — | — | 7.00 | — | — |
| SA-UNet (v1) | ICPR'20 | 82.44 | 70.15 | 83.64 | 98.19 | 96.90 | 80.85 | 98.62 | 26.54 | 0.54 | 2.29 | 1.12 |
| **SA-UNetv2 (Ours)**| **ISBI'26** | **82.82** | **70.69** | **83.64** | **98.28** | **96.98** | **81.27** | **98.71** | **21.19** | **0.26** | **1.20** | **0.95** |

#### B. With FOV Mask
| Model | Venue | F1 (%) | Jacc (%) | Sen (%) | Spe (%) | ACC (%) | MCC (%) | AUC (%) | GFLOPs | Params (M) | Mem (MB) | CPU Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IterNet | WACV'20 | 82.18 | — | 77.91 | 98.31 | 95.74 | — | 98.13 | — | 8.25 | — | — |
| RetinalLiteNet | CVPRW'24 | 80.60 | 67.50 | 78.40 | 98.00 | — | — | 97.00 | — | 0.066 | 0.25 | — |
| UNetv2† | ISBI'25 | 79.64 | — | — | — | 94.72 | — | 87.62 | — | 25.15 | — | — |
| MamUNet† | ISBI'25 | 81.78 | — | — | — | 95.36 | — | 90.25 | — | 16.86 | — | — |
| SA-UNet (v1) | ICPR'20 | 82.46 | 70.18 | 83.67 | 97.25 | 95.49 | 80.00 | 97.94 | 26.54 | 0.54 | 2.29 | 1.12 |
| **SA-UNetv2 (Ours)**| **ISBI'26** | **82.84** | **70.73** | **83.67** | **97.39** | **95.61** | **80.44** | **98.08** | **21.19** | **0.26** | **1.20** | **0.95** |

---

### Table 2: Benchmark on STARE Dataset (Without FOV Mask)
| Model | F1 (%) | Jaccard (%) | Sensitivity (%) | Specificity (%) | Accuracy (%) | MCC (%) | AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IterNet | 81.46 | — | 77.15 | 99.19 | 97.82 | — | 99.15 |
| U-Net | 79.74 | 66.43 | 83.88 | 98.45 | 97.45 | 78.77 | 98.90 |
| Attention U-Net | 80.61 | 67.72 | 84.63 | 98.49 | 97.55 | 79.60 | 98.46 |
| U-Net++ | 79.56 | 66.15 | 79.45 | 98.82 | 97.53 | 78.49 | 98.82 |
| PA-Filter | 81.70 | — | — | — | 97.88 | — | 98.43 |
| UNet 3+ | 81.16 | 68.40 | 84.50 | 98.60 | 97.60 | 80.34 | 99.06 |
| ACC-UNet-Lite | 78.99 | 65.47 | 86.12 | 98.12 | 97.24 | 78.30 | 98.85 |
| SA-UNet (v1) | 80.84 | 68.01 | **89.99** | 98.03 | 97.45 | 80.19 | **99.18** |
| **SA-UNetv2 (Ours)** | **82.81** | **70.82** | 85.35 | **98.71** | **97.83** | **81.79** | 99.13 |

---

### Table 3: Architectural Ablation Study on DRIVE (BCE Loss Only)
Shows the step-by-step contribution of each innovation:

| Configuration / Stage | Channels | F1 (%) | Jacc (%) | MCC (%) | ACC (%) | Params (M) | GFLOPs | Takeaway Insight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline SA-UNet (BN, ReLU)** | $(16,32,64,128)$ | 82.17 | 69.77 | 80.67 | 96.96 | 0.54 | 26.30 | Starting baseline |
| **2. + GroupNorm + SiLU** | $(16,32,64,128)$ | 82.49 | 70.22 | 80.92 | 96.92 | 0.54 | 26.54 | $+0.32$ F1: GN stabilizes gradients; SiLU detects low-contrast vessels |
| **3. + Channel Compression** | $(16,32,48,64)$ | 82.70 | 70.52 | 81.15 | 96.97 | **0.26** | **21.19** | Halves parameters with **improved** score (reduced overfitting) |
| **4. + Naive SA in Skips** | $(16,32,48,64)$ | 82.65 | 70.45 | 81.11 | 96.97 | 0.26 | 21.19 | Slight drop: regular SA ignores decoder, fails to bridge semantic gap |
| **5. + Full CSA Module (SA-UNetv2)**| $(16,32,48,64)$ | **82.75** | **70.60** | **81.21** | **96.99** | **0.26** | **21.19** | Best performance: bidirectional encoder-decoder interaction |

---

### Table 4: Loss Function Ablation Study on DRIVE
Tests combinations of BCE and continuous MCC ($\lambda_1 : \lambda_2$):

| Loss Composition ($\lambda_1 : \lambda_2$) | F1 (%) | Jacc (%) | Sen (%) | Spe (%) | ACC (%) | MCC (%) | AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BCE (1.0)** | 82.75 | 70.60 | 82.81 | **98.38** | **96.99** | 81.21 | 98.70 |
| **MCC (1.0)** | 82.73 | 70.56 | 84.35 | 98.16 | 96.93 | 81.17 | 91.71 |
| **BCE + MCC (0.7, 0.3)** | 82.62 | 70.41 | **84.85** | 98.07 | 96.89 | 81.06 | 98.71 |
| **BCE + MCC (0.5, 0.5) [FINAL]**| **82.82** | **70.69** | 83.64 | 98.28 | 96.98 | **81.27** | **98.71** |
| **BCE + MCC (0.3, 0.7)** | 82.76 | 70.61 | 83.81 | 98.25 | 96.96 | 81.21 | 98.68 |

> **Key takeaway:** BCE maximizes specificity; MCC boosts sensitivity for thin vessels. An exact **50/50 balance ($0.5 \text{BCE} + 0.5 \text{MCC}$)** yields optimal peak F1 ($82.82\%$), Jaccard ($70.69\%$), and MCC ($81.27\%$).

---

## 8. Ideas & Insights from Reference Paper (ACE-ProtoNet, MIA 2026)

The second paper in your workspace, *ACE-ProtoNet: Adaptive covariance eigen-gate and uncertainty-aware prototype learning for coronary artery segmentation* (Medical Image Analysis 2026), explores vessel segmentation under severe class imbalance and morphological complexity.

### Core Concepts to Draw From for Next Project Phases:
1. **Uncertainty Calibration & Awareness**:
   - Small vessels and ambiguous branching nodes often produce poorly calibrated probabilities.
   - Using predictive entropy or Monte Carlo dropout during inference allows the model to identify uncertain regions and adapt loss focus.
2. **Adaptive Covariance Eigen-Gating (ACE-Gate)**:
   - Instead of simple channel pooling, ACE-Gate uses covariance analysis and eigenvalue decomposition across feature channels to select the most discriminative cross-scale feature directions.
3. **Prototype Learning for Vessels**:
   - Maintaining dynamic foreground prototypes (vessel trunk, peripheral branches) and background prototypes (normal tissue, lesions/drusen) in latent space to prevent confusing vessels with pathological exudates.
4. **Calibration Metrics**:
   - Expected Calibration Error (ECE) and reliability diagrams can be evaluated alongside F1/Dice to measure clinical trust in vessel boundary segmentations.

---

## 9. Model Reproduction Blueprint (Checklist for Codebase)

### 9.1 Environment & Framework Compatibility
- The existing notebook uses TensorFlow/Keras:
  - `tensorflow == 2.12.0`
  - `keras_cv == 0.5.0` (for `DropBlock2D`)
  - `keras-flops` (for GFLOPs calculation)
- In modern PyTorch (if we port to PyTorch):
  - `torch.nn.GroupNorm(num_groups=8, num_channels=C)`
  - `torch.nn.SiLU()`
  - DropBlock implemented via standard PyTorch DropBlock modules.

### 9.2 Verification Milestones
- [ ] **Step 1:** Verify DRIVE dataset structure in `Drive/` (`train/images`, `train/1st_manual`, `test/images`, `test/1st_manual`, `test/mask`).
- [ ] **Step 2:** Instantiate SA-UNetv2 architecture with input shape $(592, 592, 3)$, channel progression $(16, 32, 48, 64)$, GroupNorm ($G=8$), SiLU, DropBlock ($rate=0.15, block\_size=7$).
- [ ] **Step 3:** Verify parameter count is exactly $\approx \mathbf{0.26\text{M}}$ (260,000–265,000 trainable weights).
- [ ] **Step 4:** Implement continuous MCC Loss and composite $0.5 \times \text{BCE} + 0.5 \times \text{MCC}$.
- [ ] **Step 5:** Run training with Adam ($10^{-3}$), batch size 8, ReduceLROnPlateau, EarlyStopping (patience 20).
- [ ] **Step 6:** Run test evaluation with threshold $0.5$, both with and without FOV mask, targeting $\mathbf{F1 \approx 82.8\%}$ and $\mathbf{Jaccard \approx 70.7\%}$.
