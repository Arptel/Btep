# Proposed Research & Model Edits Workspace

This folder is your **dedicated experimental workspace** for developing novel enhancements beyond the baseline paper.

All changes here are completely isolated from `baseline_reproduction/`, so your verified baseline reproduction remains safe and untouched.

---

## 📁 Folder Contents

* **`src/`**: Your editable modular codebase:
  * `model.py`: Modify this or add new architectures (e.g. ACE-Gate, dynamic prototypes, attention variants).
  * `losses.py`: Modify this or add new loss functions (e.g. clDice for topological continuity, focal loss, uncertainty loss).
  * `dataset.py`: Shared DRIVE dataset pipeline (reads automatically from `../Drive`).
  * `stare_dataset.py`: Shared STARE dataset pipeline (reads automatically from `../Stare data`).
  * `train.py`: Training engine ready for novel loss functions and schedulers.
  * `evaluate.py`: Evaluation engine.
* **`reproduce.py`**: Training runner for DRIVE experiments.
* **`train_stare.py`**: Training runner for STARE experiments.
* **`evaluate_test.py`**: Test evaluation script.
* **`checkpoints/`**: Dedicated directory where your new experimental model weights will be saved.
* **`results/`**: Dedicated directory for your experimental predictions and visualizations.

---

## 🎯 Planned Novel Directions (Ideas for Edits)

1. **Topology-Preserving Loss (clDice)**:
   * Penalizes broken vessel skeletons and improves connectivity in fine capillaries.
2. **Adaptive Covariance Eigen-Gating (ACE-Gate)**:
   * Replaces static channel pooling in skip connections with covariance-informed feature filtering.
3. **Prototype Learning for Retinal Vessels**:
   * Learns latent cluster prototypes to distinguish true vessels from pathological drusen and exudates.
4. **Uncertainty Calibration (ECE)**:
   * Evaluates Expected Calibration Error and confidence maps for clinical reliability.

---

## 🚀 Running Experiments

From the `edits/` directory:

```powershell
# 1. Navigate to edits
cd edits

# 2. Run your experimental training
py reproduce.py

# 3. Evaluate your experimental checkpoint
py evaluate_test.py
```
