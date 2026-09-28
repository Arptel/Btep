# SA-UNetv2 Baseline Reproduction

This folder contains the complete, self-contained implementation reproducing the **SA-UNetv2** paper (*ISBI 2026 Oral Presentation*):
> **"SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation"**  
> *Changlu Guo, Anders Nymark Christensen, Anders Bjorholm Dahl, Yugen Yi, Morten Rieger Hannemose*

---

## 📁 Folder Contents

* **`src/`**: Core modules for the baseline reproduction:
  * `model.py`: SA-UNetv2 architecture with DropBlock, Group Normalization ($G=8$), SiLU, and Cross-scale Spatial Attention (CSA).
  * `dataset.py`: DRIVE dataset loader, zero-padding ($584 \times 565 \to 592 \times 592$), and augmentations.
  * `stare_dataset.py`: STARE dataset loader, zero-padding ($605 \times 700 \to 704 \times 704$), and augmentations.
  * `losses.py`: Continuous Matthews Correlation Coefficient (MCC) loss and compound loss ($\mathcal{L} = 0.5 \text{BCE} + 0.5 \text{MCC}$).
  * `train.py`: Training engine with Adam optimizer and ReduceLROnPlateau.
  * `evaluate.py`: DRIVE evaluation engine (computes metrics with and without FOV mask).
* **`reproduce.py`**: End-to-end DRIVE reproduction script (trains for 150 epochs and evaluates on test set).
* **`train_stare.py`**: STARE dataset training pipeline.
* **`evaluate_test.py`**: Standalone evaluation script for DRIVE test set using reproduced weights.
* **`evaluate_stare.py`**: Standalone evaluation script for STARE test set (generates glowing overlays and visual comparisons).
* **`infer.py`**: Single-image visual inference script with neon green overlay generation.

---

## 🚀 Quick Execution Commands

From the repository root or from inside this folder:

### 1. Evaluate DRIVE Baseline (20 Test Images)
```powershell
py baseline_reproduction/evaluate_test.py
```

### 2. Evaluate STARE Baseline (Test Set)
```powershell
py baseline_reproduction/evaluate_stare.py
```

### 3. Run Inference on a Single Image
```powershell
py baseline_reproduction/infer.py --image "Drive/DRIVE/test/images/01_test.tif" --output "results/01_test_pred.png"
```

*Note: Datasets (`Drive/` and `Stare data/`), trained checkpoints (`checkpoints/`), and visual outputs (`results/`) remain at the project root for unified access.*
