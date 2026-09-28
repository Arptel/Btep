# Idea 1: Topology-Preserving clDice Loss (Soft-clDice)

**Research Objective:**  
Eliminate retinal vessel fragmentation and boost micro-capillary connectivity by incorporating a differentiable **soft-clDice loss** into the compound training objective.

---

## 🔬 Motivation from Our Baseline Findings

Our baseline evaluation exposed severe topological fragmentation in the original SA-UNetv2 model:
* **Vascular Fragmentation:** **$43.01\times$** on DRIVE (Ground Truth has 3 connected components; Baseline predicts **82 disjoint fragments**).
* **Topology Sensitivity ($T_{\text{sens}}$):** Only **$71.09\%$** (the baseline misses nearly **$29\%$** of all true vessel centerlines).

---

## 🛠️ Formulation

Standard BCE and MCC treat all pixels independently. In this experiment, we introduce:
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{BCE}} + \lambda_2 \mathcal{L}_{\text{MCC}} + \lambda_3 \mathcal{L}_{\text{soft-clDice}}$$

Where soft-clDice uses differentiable morphological erosion (min-pooling) to extract continuous vessel centerlines directly inside PyTorch autograd.

---

## 📁 Workspace Structure

* **`src/`**:
  * `losses.py`: Implements `SoftclDiceLoss` and `TopologicalCompoundLoss`.
  * `model.py`: SA-UNetv2 backbone.
  * `dataset.py` & `stare_dataset.py`: Shared data loaders.
  * `train.py`: Training engine supporting the new topological loss.
* **`reproduce.py`**: Training runner for DRIVE dataset with clDice.
* **`train_stare.py`**: Training runner for STARE dataset with clDice.
* **`evaluate_test.py`**: Standard pixel evaluation (Dice, Jaccard, MCC).
* **`evaluate_connectivity.py`**: Topological evaluation (clDice, Betti-0 fragmentation, LCCR).
* **`checkpoints/`**: Independent weights storage (will NOT overwrite baseline).
* **`results/`**: Independent visual predictions and overlays.
