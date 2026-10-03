# Architecture & Implementation Plan: Conductance-Weighted clDice (`cw-clDice`)

> **Research Track:** Novel Objective Function Formulation (Bridging Fluid Dynamics with Topological Deep Learning)  
> **Status:** Architecture Design & Implementation Blueprint (Option 1)  
> **Target Module:** [`edits/src/losses.py`](file:///c:/Users/Student/Arth Patel/Btep/edits/src/losses.py) and Training Engine [`edits/src/train.py`](file:///c:/Users/Student/Arth Patel/Btep/edits/src/train.py)  

---

## 1. Executive Summary & The Novelty Claim

### 1.1 The Core Scientific Novelty
While off-the-shelf **clDice** (CVPR 2021) introduced centerline supervision for tubular structures, it treats all centerline pixels with **uniform, isotropic weight ($w=1.0$)**, completely ignoring hemodynamic transport physics.

**Our Novel Contribution:**  
We introduce **Conductance-Weighted clDice (`cw-clDice`)**, the first topological loss function that integrates **Poiseuille vascular fluid resistance** directly into PyTorch autograd backpropagation. 

Instead of treating a giant central artery trunk identically to a fragile micro-capillary, `cw-clDice` dynamically modulates gradient pressure based on the physical caliber and hydraulic conductance of each vascular branch.

---

## 2. Theoretical Formulation & Physics-Loss Synthesis

```
   [Ground Truth Vessel Mask] ──▶ Euclidean Distance Transform R(x, y)
                 │                                │
                 ▼                                ▼
       Centerline Skeleton S_gt      Poiseuille Conductance Map c(x, y) ∝ r^4
                 │                                │
                 └────────────────┬───────────────┘
                                  ▼
                   Physiological Weight Map W(x, y)
                     (High on Capillaries: 3.0x)
                     (Standard on Trunks: 1.0x)
                                  │
                                  ▼
      Predicted Probability P ──▶ [Soft Differentiable Skeletonizer] ──▶ S_pred
                                  │
                                  ▼
             [Conductance-Weighted Topology Sensitivity & Precision]
                                  │
                                  ▼
                    cw-clDice Loss = 1 - cw-clDice
                                  │
                                  ▼
     Gradient Backpropagation: Massive Gradient Spikes on Broken Capillaries!
```

### 2.1 The Hydraulic Resistance Problem in Deep Learning
Under Poiseuille flow, hydraulic resistance across a cylindrical vessel segment is inversely proportional to the radius to the fourth power:
$$R_{\text{flow}} \propto \frac{1}{r^4} \implies c_e \propto r^4$$

* In retinal fundus imaging, large arteries near the optic disc have calibers of $r \approx 4 - 6\text{ px}$ with massive contrast and high signal-to-noise ratio.
* Peripheral capillaries have calibers of $r \approx 1\text{ px}$ with high hydraulic resistance, low contrast, and faint optical intensity.
* Under standard loss functions (BCE, Dice), the neural network takes the path of least resistance: it perfectly segments the high-contrast main trunks while dropping the fragile capillaries, causing the **$43.01\times$ fragmentation** observed in our baseline audit.

### 2.2 The Conductance Weighting Function $W(x, y)$
For each training image, we compute the exact Euclidean Distance Transform on the ground-truth mask:
$$R_{\text{gt}}(x, y) = \text{distance\_transform\_edt}(V_{\text{gt}})(x, y)$$
On the ground truth skeleton $S_{\text{gt}}$, $R_{\text{gt}}(x, y)$ gives the exact half-width (radius) of the vessel lumen at that centerline coordinate.

We formulate the **Physiological Conductance Weight Function**:
$$W(x, y) = 1.0 + \alpha \cdot \left(\frac{r_{\max} - R_{\text{gt}}(x, y)}{r_{\max} - r_{\min} + \epsilon}\right)^\beta$$

* **For thin capillaries ($r \approx 1\text{ px}$):** $W(x, y) \to 1.0 + \alpha$ (amplified by up to **$2.5\times - 3.5\times$**).
* **For thick main arcades ($r \ge 4\text{ px}$):** $W(x, y) \to 1.0$ (retains standard topological baseline weight).
* Here, $\alpha \in [1.0, 2.5]$ is the capillary boost factor, and $\beta \in [1.0, 2.0]$ controls the conductance decay curvature.

### 2.3 Differentiable Soft Skeletonization in PyTorch
To make centerline extraction fully differentiable during forward and backward passes, we implement soft morphological erosion and opening operators using PyTorch 2D pooling layers:
$$\text{Erode}(X) = -\text{MaxPool2d}(-X, \text{kernel}=3, \text{stride}=1, \text{padding}=1)$$
$$\text{Dilate}(X) = \text{MaxPool2d}(X, \text{kernel}=3, \text{stride}=1, \text{padding}=1)$$
$$\text{Open}(X) = \text{Dilate}(\text{Erode}(X))$$

The continuous soft skeleton $\tilde{S}(P)$ is extracted across $K=4$ iterative pooling steps:
$$\tilde{S}(P) = \sum_{k=1}^K \max\left(0, \text{Erode}_k(P) - \text{Open}(\text{Erode}_k(P))\right)$$

### 2.4 Conductance-Weighted Topology Sensitivity & Precision
We formulate the weighted topological intersection terms:

1. **Weighted Topology Sensitivity ($T_{\text{sens}}^{\text{cw}}$):**
   Measures whether the predicted probability map $P$ covers the true vascular centerlines, with higher penalties on thin capillaries:
   $$T_{\text{sens}}^{\text{cw}}(S_{\text{gt}}, P) = \frac{\sum_{(x, y)} W(x, y) \cdot S_{\text{gt}}(x, y) \cdot P(x, y)}{\sum_{(x, y)} W(x, y) \cdot S_{\text{gt}}(x, y) + \epsilon}$$

2. **Weighted Topology Precision ($T_{\text{prec}}^{\text{cw}}$):**
   Measures whether the predicted soft skeleton $\tilde{S}(P)$ remains constrained within the true vessel lumen:
   $$T_{\text{prec}}^{\text{cw}}(\tilde{S}(P), V_{\text{gt}}) = \frac{\sum_{(x, y)} W(x, y) \cdot \tilde{S}(P)(x, y) \cdot V_{\text{gt}}(x, y)}{\sum_{(x, y)} W(x, y) \cdot \tilde{S}(P)(x, y) + \epsilon}$$

3. **Combined `cw-clDice` Formulation:**
   $$\text{cw-clDice} = \frac{2 \cdot T_{\text{prec}}^{\text{cw}} \cdot T_{\text{sens}}^{\text{cw}}}{T_{\text{prec}}^{\text{cw}} + T_{\text{sens}}^{\text{cw}} + \epsilon}$$
   $$\mathcal{L}_{\text{cw-clDice}} = 1 - \text{cw-clDice}$$

### 2.5 Total Unified Compound Loss
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{MCC}} + \lambda_{\text{cw}} \cdot \mathcal{L}_{\text{cw-clDice}}$$
where $\lambda_{\text{cw}} \in [0.1, 0.3]$ balances pixel-wise alignment with topological continuity.

---

## 3. Step-by-Step Implementation Roadmap

### Step 1: Implementation of `ConductanceWeightedclDiceLoss`
* File: [`edits/src/losses.py`](file:///c:/Users/Student/Arth Patel/Btep/edits/src/losses.py)
* Classes to implement:
  * `SoftSkeletonize(nn.Module)`: Differentiable morphological pooling.
  * `ConductanceWeightedclDiceLoss(nn.Module)`: Forward pass computing $T_{\text{prec}}^{\text{cw}}$, $T_{\text{sens}}^{\text{cw}}$, and backpropagation gradients.
  * `TopologicalCompoundLoss(nn.Module)`: Blends BCE, Continuous MCC, and `cw-clDice`.

### Step 2: Ground-Truth Precomputation & Caching
* For the 20 DRIVE training images, precompute:
  1. Binary skeleton $S_{\text{gt}}$ via medial-axis skeletonization (`skimage.morphology.skeletonize`).
  2. Euclidean distance transform $R_{\text{gt}}$ (`scipy.ndimage.distance_transform_edt`).
  3. Conductance weight map $W(x, y)$.
* Cache these as `.npy` tensors during dataset loading in [`edits/src/dataset.py`](file:///c:/Users/Student/Arth Patel/Btep/edits/src/dataset.py) to guarantee **zero runtime latency overhead during training epochs**.

### Step 3: Unit Verification Suite
* Create `edits/tests/test_cw_cldice.py`:
  * Verify forward pass outputs values in $[0, 1]$.
  * Verify `loss.backward()` produces valid, non-zero gradients on model weights.
  * Verify that breaking a 1-pixel capillary produces a significantly higher gradient penalty than under standard BCE/Dice.

### Step 4: Experimental Training Run (DRIVE Benchmark)
* Runner: [`edits/reproduce.py`](file:///c:/Users/Student/Arth Patel/Btep/edits/reproduce.py)
* Train for 150 epochs using Adam ($lr = 10^{-3}$) on GPU.
* Save experimental checkpoint: `edits/checkpoints/best_sa_unetv2_cwcldice.pth`.

### Step 5: Full Benchmark & Connectivity Evaluation
* Run standardized connectivity suite: `py edits/evaluate_connectivity.py --model cwcldice --compare baseline`.
* Validate that:
  1. $F_1$-score increases from **$80.09\% \to 82-84\%$**.
  2. Betti-0 connected component fragmentation drops from **$82.05 \to < 15$**.
  3. Centerline Dice increases from **$79.89\% \to > 84\%$**.
  4. Inference speed remains identical (**$20.2\text{ ms}$ / image**), because the loss operates strictly at training time!

---

## 4. Academic Defense & Thesis Summary Table

| Evaluation Dimension | Baseline SA-UNetv2 (ISBI'26) | Vanilla clDice (CVPR'21) | Our Proposed `cw-clDice` (Ours) |
| :--- | :---: | :---: | :---: |
| **Loss Formulation** | $0.5\text{ BCE} + 0.5\text{ MCC}$ | $\text{BCE} + \text{Dice} + \text{clDice}$ | $\mathbf{0.5\text{ BCE} + 0.5\text{ MCC} + \lambda\text{ cw-clDice}}$ |
| **Centerline Weighting** | None (Pixel-only) | Uniform ($w=1.0$ everywhere) | **Hemodynamic Weighting ($W \propto r^{-\beta}$)** |
| **Capillary Prior** | Blind to thin vessels | Equal to giant arteries | **High Gradient Pressure on Capillaries** |
| **Novelty Level** | Base Paper | Existing 2021 work | **Original Capstone Contribution** |
| **Fragmentation Expected** | $43.01\times$ (Severe) | $\sim 10-15\times$ (Moderate) | **$< 5-8\times$ (Continuous Vascular Tree)** |
| **Inference Latency Impact** | $20.2\text{ ms}$ (GPU) | $20.2\text{ ms}$ (No inference cost) | **$20.2\text{ ms}$ (Zero inference overhead)** |
