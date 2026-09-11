# SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation (ISBI 2026, Oral Presentation)

Official implementation of **SA-UNetv2**, a lightweight and high-precision deep learning model for retinal vessel segmentation. This work has been accepted as an **Oral Presentation** at **ISBI 2026**.

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://clguo.github.io/SA-UNetv2)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2509.11774)

## 🚀 Highlights
SA-UNetv2 improves upon the original SA-UNet by addressing class imbalance and multi-scale feature fusion gaps. It is designed for clinical deployment in resource-constrained, CPU-only environments.

- **Ultra-Lightweight**: Only **0.26M** parameters and **1.2MB** model size (50% smaller than SA-UNet).
- **Efficient**: Sub-second inference (**0.95s**) on a standard CPU.
- **State-of-the-Art**: Achieves top-tier performance on DRIVE and STARE datasets.

## 🛠️ Key Innovations
1. **Cross-scale Spatial Attention (CSA)**: Integrated into all skip connections to bridge the semantic gap between encoder and decoder features for the first time.
2. **Optimized Convolutional Blocks**: Redesigned with **Group Normalization (GN)** and **SiLU** activation for enhanced stability and gradient flow during training.
3. **BCE + MCC Compound Loss**: Combines pixel-level accuracy (BCE) with global correlation (MCC) to effectively handle extreme foreground-background imbalance.

## 📊 Performance
Comparison on the DRIVE dataset (Without FOV Masking):

| Model | F1 (%) | Jaccard (%)| MCC (%) | Params (M) | CPU Inference (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| U-Net (MICCAI'15) | 81.30 | 68.52 | 79.66 | 8.64 | 3.16 |
| Attention U-Net (MIDL'18) | 81.47 | 68.76 | 79.84 | 8.65 | 7.74 |
| SA-UNet (ICPR'20) | 82.44 | 70.15 | 80.85 | 0.54 | 1.12 |
| **SA-UNetv2 (Ours)** | **82.82** | **70.69** | **81.27** | **0.26** | **0.95** |


## 📝 Citation
If you find this work or code helpful, please cite our paper:
```bibtex
@inproceedings{guo2025saunetv2,
  author    = {Guo, Changlu and Christensen, Anders Nymark and Dahl, Anders Bjorholm and Yi, Yugen and Hannemose, Morten Rieger},
  title     = {SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation},
  booktitle = {Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026},
}
```
