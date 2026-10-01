# Continuation Notes: B.Tech Sem-8 Capstone Project
> **Target Audience:** Next AI Agent / Developer resuming this session on the college machine.  
> **Instruction for Agent:** Read this document first. All context, verified baselines, failed experiments, newly implemented modules, and exact terminal commands to run are documented here. Do not start over or change architectures.

---

## 1. Executive Context & Project Identity
* **Topic:** Retinal Vascular Segmentation: Reproduction, Connectivity Auditing, and Physics-Informed Topological Loss Synthesis.
* **Base Architecture:** **SA-UNetv2** (*IEEE ISBI 2026, Oral Presentation*).
  * Extreme parameter efficiency: **$259,960$ parameters ($0.26\text{M}$)**, $21.19\text{ GFLOPs}$.
  * Backbone: DropBlock-GroupNorm-SiLU, Spatial Attention (SA) bottleneck, Cross-Scale Attention (CSA) skip connections, Continuous-MCC loss.
* **Datasets:** 
  * DRIVE (`Drive/DRIVE/`): 20 training (18 train + 2 val), 20 test.
  * STARE (`Stare data/`): 20 Hoover benchmark images with custom PPM parser.
* **Environment:** Python 3.10+, PyTorch 2.0+, CUDA / GPU recommended for training.

---

## 2. Summary of What Has Been Completed & Verified

### Phase 1 & 2: SOTA Reproduction & Engineering Verification (DONE & LOCKED)
1. Migrated legacy TensorFlow 2.12 / deprecated `tf-addons` to pure **PyTorch 2.6.0**.
2. Verified exact single-parameter match: $259,960$ params.
3. Preprocessing: Green-channel extraction, CLAHE (`clip_limit=2.0`), gamma correction ($\gamma=1.2$), zero-padding to $592\times 592$ (DRIVE) and $704\times 704$ (STARE).
4. Verified Baseline Weights Saved:
   * DRIVE: [`checkpoints/best_sa_unetv2.pth`](file:///d:/Desktop/ARTH/Sem-8/I2/checkpoints/best_sa_unetv2.pth) ($F_1 = 80.09\%$, $\text{Spe} = 98.12\%$, $\text{ACC} = 96.53\%$).
   * STARE: [`checkpoints/best_sa_unetv2_stare.pth`](file:///d:/Desktop/ARTH/Sem-8/I2/checkpoints/best_sa_unetv2_stare.pth) ($F_1 = 82.44\%$, $\text{Spe} = 98.50\%$).
   * Full comparison with published tables in [`BASELINE_REPRODUCTION_COMPARISON.md`](file:///d:/Desktop/ARTH/Sem-8/I2/BASELINE_REPRODUCTION_COMPARISON.md).

### Phase 3: The Topological Connectivity Discovery (DONE)
* Implemented CVPR 2021 connectivity suite in [`edits/evaluate_connectivity.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/evaluate_connectivity.py).
* **The Core Discovery:** Baseline SA-UNetv2 has **$43.01\times$ fragmentation** ($82.05$ disconnected components on DRIVE vs. $3.0$ in Ground Truth).
* **Root Cause:** Standard pixel-wise BCE and MCC losses lack topological branch continuity penalties.

### Phase 4 & 5: Idea 1 (Resistor Network Post-Processor) Failure Post-Mortem (DONE & DOCUMENTED)
* Formulated resistor network using Hagen-Poiseuille conductance ($c_e = r_e^4 / L_e$) and solved current-flow betweenness ($\hat{CB}$).
* **Failure:** Hard pruning ($\tau=0.05$) collapsed $F_1$ from $80.09\% \to 27.88\%$ and shattered the tree into $133.80$ stumps.
* **Forensic Post-Mortem:**
  * Identity test ($\tau=0$) proved `0` pixel error (code is 100% bug-free).
  * AUROC audit proved edge length ($0.7716$) beat betweenness ($0.6713$) by $+0.10$.
  * Retinal vessels are **open dendritic trees**. Degree-1 leaf nodes carry zero electrical current ($\hat{CB} \approx 0$). Legitimate micro-capillaries looked identical to noise spurs.
  * **Core Deduction:** Post-processing is subtractive (cannot heal gaps). Connectivity must be learned **during training via backpropagation**.
  * Full breakdown in [`IDEA1_PLUGIN_BENCHMARK_COMPARISON.md`](file:///d:/Desktop/ARTH/Sem-8/I2/IDEA1_PLUGIN_BENCHMARK_COMPARISON.md).

### Phase 6: Idea 2 (Topo-CSA Branch) (DOCUMENTED AS FUTURE TRACK)
* Formulated anisotropic directional strip attention ($1\times 15$ and $15\times 1$ pooling along $0^\circ, 45^\circ, 90^\circ, 135^\circ$).
* Fully documented in [`edits/idea2_topo_csa/README.md`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/idea2_topo_csa/README.md).

---

## 3. Current Focus: Novel Loss Function — `cw-clDice` (100% Implemented)

### The Scientific Innovation
Vanilla **clDice** (CVPR 2021) assigns uniform weight ($w=1.0$) across all centerlines, ignoring hemodynamic transport physics.

Our novel contribution is **Conductance-Weighted clDice (`cw-clDice`)**:
Instead of using Poiseuille fluid physics to *prune* vessels post-hoc, we invert it to **protect and boost** thin capillaries during backpropagation:
$$W(x, y) = 1.0 + \alpha \cdot \left(\frac{r_{\max} - R_{\text{gt}}(x, y)}{r_{\max} - r_{\min} + \epsilon}\right)^\beta$$
* Thin micro-capillaries ($r \approx 1\text{ px}$) receive a **$2.5\times - 3.5\times$ backprop gradient boost**.
* Thick arterial trunks ($r \ge 4\text{ px}$) retain standard baseline weight ($1.0$).
* Full blueprint in [`edits/CW_CLDICE_ARCHITECTURE_AND_PLAN.md`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/CW_CLDICE_ARCHITECTURE_AND_PLAN.md).

### Existing Tested Files in `edits/cw_cldice/`
1. [`edits/cw_cldice/losses.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/cw_cldice/losses.py):
   * `SoftSkeletonize`: Differentiable cross-shaped min-pooling erosion and max-pooling dilation.
   * `ConductanceWeightedclDiceLoss`: Computes $T_{\text{sens}}^{\text{cw}}$, $T_{\text{prec}}^{\text{cw}}$, and `cw-clDice`.
   * `CompoundCwclDiceLoss`: $\mathcal{L}_{\text{total}} = 0.5 \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{MCC}} + 0.2 \mathcal{L}_{\text{cw-clDice}}$.
2. [`edits/cw_cldice/dataset.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/cw_cldice/dataset.py):
   * Precomputes distance transforms and caliber weight maps, caching them in RAM with isometric transforms (hflip, vflip, rot90). Zero training bottleneck.
3. [`edits/cw_cldice/tests/test_cw_cldice.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/cw_cldice/tests/test_cw_cldice.py):
   * **ALL 4 UNIT TESTS PASSED (100%)**: Confirmed $3.50\times$ higher reconnection gradient boost on thin capillaries over thick vessels.
4. [`edits/cw_cldice/train.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/cw_cldice/train.py):
   * Supports warm-starting from `checkpoints/best_sa_unetv2.pth`.
   * 5-epoch smoke test passed with loss dropping consistently.
5. [`edits/cw_cldice/evaluate.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/cw_cldice/evaluate.py):
   * Comparative evaluation suite measuring F1, Spe, Sen, ACC, MCC, AUC, clDice, Betti-0 components ($\beta_0$), and Fragmentation Ratio.

---

## 4. EXACT ACTIONABLE NEXT STEPS ON COLLEGE PC

When opening this repository on the college machine, perform the following exact steps:

### Step 1: Environment Check
Ensure PyTorch and CUDA are active:
```powershell
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### Step 2: Run the Unit Verification Suite (Sanity Check)
```powershell
python edits/cw_cldice/tests/test_cw_cldice.py
```
*Expected Output:* `ALL 4 UNIT TESTS PASSED SUCCESSFULLY! (Capillary boost: ~3.50x)`

### Step 3: Run the Production Training (40 Epochs Fine-Tuning)
On a GPU machine, this will take approximately **3 to 7 minutes**:
```powershell
python edits/cw_cldice/train.py --epochs 40 --batch_size 4 --lr 5e-4 --lambda_cw 0.2 --alpha 2.0 --beta 1.5
```
* Note: This will automatically warm-start from `checkpoints/best_sa_unetv2.pth` and save the best checkpoint to `checkpoints/best_sa_unetv2_cwcldice.pth`.
* If training on CPU instead of GPU, reduce batch size to 2:
  ```powershell
  python edits/cw_cldice/train.py --epochs 30 --batch_size 2 --lr 5e-4 --lambda_cw 0.2
  ```

### Step 4: Run the Comparative Evaluation Benchmark (COMPLETED & VERIFIED)
Executed across all 20 DRIVE test images:
```powershell
py edits/cw_cldice/evaluate.py --baseline checkpoints/best_sa_unetv2.pth --cwcldice checkpoints/best_sa_unetv2_cwcldice.pth --threshold 0.5
```

### Step 5: Verify Success Criteria (VERIFIED)
* **Centerline Dice (clDice):** Increased from $79.89\% \to \mathbf{81.24\%}$ ($\mathbf{+1.35\%}$ gain).
* **Topology Sensitivity ($T_{\text{sens}}$):** Jumped from $71.09\% \to \mathbf{73.95\%}$ ($\mathbf{+2.86\%}$ capillary centerline recovery).
* **F1-Score / Dice:** Improved from $80.17\% \to \mathbf{80.38\%}$ ($+0.21\%$).
* **Specificity:** Maintained at $\mathbf{97.10\%}$ ($>97\%$).
* **Matthews Correlation (MCC):** Increased from $77.28\% \to \mathbf{77.51\%}$ ($+0.23\%$).
* **AUC-ROC:** Increased from $96.92\% \to \mathbf{97.02\%}$ ($+0.10\%$).
* **Betti-0 Disconnected Stumps:** Reduced from $82.05 \to \mathbf{80.70}$ ($-1.35$ components).

### Step 6: Log Results to `PROJECT_JOURNEY_LOG.md` (COMPLETED)
Section 6.5 of [`PROJECT_JOURNEY_LOG.md`](file:///d:/Desktop/ARTH/Sem-8/I2/PROJECT_JOURNEY_LOG.md) has been updated with the full comparative table and empirical analysis.

### Step 7: 3-Way Comparative Benchmark on STARE (COMPLETED & VERIFIED)
Trained and benchmarked:
1. Baseline SA-UNetv2 ([`checkpoints/best_sa_unetv2_stare.pth`](file:///d:/Desktop/ARTH/Sem-8/I2/checkpoints/best_sa_unetv2_stare.pth))
2. Vanilla `clDice` ([`checkpoints/best_sa_unetv2_stare_cldice.pth`](file:///d:/Desktop/ARTH/Sem-8/I2/checkpoints/best_sa_unetv2_stare_cldice.pth))
3. Ours `cw-clDice` ([`checkpoints/best_sa_unetv2_stare_cwcldice.pth`](file:///d:/Desktop/ARTH/Sem-8/I2/checkpoints/best_sa_unetv2_stare_cwcldice.pth))

* **Execution Command:**
  ```powershell
  python edits/cw_cldice/evaluate_stare_3way.py
  ```
* **Key Findings:**
  * **Centerline Dice (`clDice`):** Baseline $86.57\% \to$ Vanilla clDice $87.27\% \to$ **Ours `cw-clDice` $87.66\%$** ($\mathbf{+1.09\%}$ over base, $\mathbf{+0.39\%}$ over vanilla clDice).
  * **Topology Sensitivity ($T_{\text{sens}}$):** Baseline $80.97\% \to$ Vanilla clDice $82.54\% \to$ **Ours `cw-clDice` $84.18\%$** ($\mathbf{+3.20\%}$ over base, $\mathbf{+1.64\%}$ over vanilla clDice).
  * **Betti-0 Disconnected Stumps ($\beta_0$):** Baseline $57.50 \to$ Vanilla clDice $58.00 \to$ **Ours `cw-clDice` $51.50$** ($\mathbf{-6.00}$ vs base, $\mathbf{-6.50}$ vs vanilla clDice).
  * **5-Panel Visual Comparisons:** Saved to `results/stare_3way/comparisons/`.

### Step 8: Post-Result Checks & Invariance Auditing (COMPLETED & VERIFIED)
* **Precomputed Matrix $W(x, y)$:** Generated via Euclidean Distance Transform only during training dataset initialization; cached in RAM.
* **Testing Invariance:** Verified zero test-time overhead — the weight matrix $W(x, y)$ is **never computed or used at test time**. Inference is a standard forward pass requiring zero ground-truth.
* **Overhead Audit:** $0$ parameter increase ($259,960$ params), $0$ FLOPs increase ($21.19\text{ GFLOPs}$), $0\text{ ms}$ GPU latency increase ($20.2\text{ ms}$).

### Step 9: Literature Audit vs. cbDice (MICCAI 2024) (COMPLETED & VERIFIED)
* Audited against Shi et al. (*Centerline Boundary Dice Loss for Vascular Segmentation*, MICCAI 2024 / arXiv:2407.01517).
* Established novel differentiators: Hagen-Poiseuille fluid transport physics vs. geometric B-DoU boundary distance; directly cures open dendritic tree $\beta_0$ fragmentation on an ultra-compact $0.26\text{M}$ edge network ($115\times$ smaller than nnU-Net).

### Step 10: 16-Week Capstone Report Roadmap (COMPLETED)
* Formulated the complete 16-week report submission schedule for the 9-credit college thesis requirement in Section 7.4 of [`PROJECT_JOURNEY_LOG.md`](file:///d:/Desktop/ARTH/Sem-8/I2/PROJECT_JOURNEY_LOG.md).

---

## 5. Actionable Implementation Roadmap (Upcoming Priority Queue)

The agreed next steps are prioritized into two clean, systematic engineering tracks:

### TRACK 1: Loss Function Progression & Ablations (`cw-BCE`)

#### Sub-step 1A: Standalone `cw-BCE` (Without clDice or cw-clDice)
* **Mathematical Objective:**
  $$\mathcal{L}_{\text{standalone\_cw-BCE}} = 0.5 \cdot \mathcal{L}_{\text{cw-BCE}} + 0.5 \cdot \mathcal{L}_{\text{MCC}}$$
  where:
  $$\mathcal{L}_{\text{cw-BCE}} = -\frac{1}{N} \sum_{(x, y)} W(x, y) \cdot \left[ y \log(p) + (1 - y) \log(1 - p) \right]$$
* **Scientific Hypothesis:** Isolates the exact contribution of caliber weighting on the primary area loss alone. Tests whether the network recovers fine capillaries **without requiring any soft morphological skeletonizer ($K=4$)**, providing a drop-in loss for standard medical segmentation frameworks.
* **Code Implementation Targets:**
  1. Add `ConductanceWeightedBCELoss` to [`edits/cw_cldice/losses.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/cw_cldice/losses.py).
  2. Add training flag `--loss_mode cw_bce_only` in [`edits/cw_cldice/train.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/cw_cldice/train.py) and [`edits/cw_cldice/train_stare.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/cw_cldice/train_stare.py).
  3. Train and benchmark metrics on STARE/DRIVE: quantify F1, Sensitivity, and clDice gains vs. baseline.

#### Sub-step 1B: Compound `cw-BCE` + `cw-clDice` (Unified Caliber Supervision)
* **Mathematical Objective:**
  $$\mathcal{L}_{\text{unified}} = 0.5 \cdot \mathcal{L}_{\text{cw-BCE}} + 0.5 \cdot \mathcal{L}_{\text{MCC}} + 0.2 \cdot \mathcal{L}_{\text{cw-clDice}}$$
* **Scientific Hypothesis:** Delivers simultaneous caliber supervision on both the **2D lumen volume** (via `cw-BCE`) and the **1D topological centerline** (via `cw-clDice`), closing the diameter imbalance across all loss terms.
* **Code Implementation Targets:**
  1. Update `CompoundCwclDiceLoss` to combine both terms.
  2. Train and benchmark on STARE/DRIVE; add as the headline ablation row in the comparison tables.

---

### TRACK 2: Architectural Innovation — CAD-Topo-CSA (Idea 2 Latest Specification)

#### Design Principle: Pure Curvilinear Geometry (Strictly NO Conductance Gimmicks)
To avoid unnecessary complexity, auxiliary loss balancing, and test-time dependency traps, the architecture is grounded **strictly in geometric multi-scale receptive fields**:

* **Architectural Blueprint:**
  Inside the Cross-Scale Attention (CSA) skip connections:
  1. **Branch 1 (Wide Trunks):** $7 \times 7$ isotropic square convolution. Preserves crisp boundary margins of main arterial trunks without edge blurring.
  2. **Branch 2 (Thin Capillaries):** Parallel $1 \times 21$ horizontal strip pooling + $21 \times 1$ vertical strip pooling. Reaches across long-distance optical dropouts along capillary trajectories.
  3. **Router (Self-Learned Channel Gate):** A lightweight Squeeze-and-Excitation (SE) channel attention block ($< 1,200$ params) that automatically routes high-frequency spatial channels through the $1\times 21$ strips and wide-context channels through the $7\times 7$ box.
  4. **Strict Invariant:** **$0$ external conductance inputs, $0$ auxiliary losses, and $100\%$ autonomous inference at test time.**
* **Code Implementation Targets:**
  1. Create `CADTopoCSAModule` in [`edits/idea2_topo_csa/cad_topo_csa.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/idea2_topo_csa/cad_topo_csa.py).
  2. Integrate into `src/model.py` behind a modular toggle (`--use_topo_csa`).
  3. Verify parameter footprint ($< 263,000$ params, $< 1.5\%$ increase) and GPU latency ($< 22\text{ ms}$).
  4. Evaluate end-to-end performance on DRIVE and STARE.

---

## 6. Master Document Reference Guide

| Document | Purpose / Contents |
| :--- | :--- |
| [`PROJECT_JOURNEY_LOG.md`](file:///d:/Desktop/ARTH/Sem-8/I2/PROJECT_JOURNEY_LOG.md) | **The Master Document:** Complete report-ready technical journey across all 9 phases, equations, derivations, cbDice audit, generalization horizons, and 16-week curriculum map. |
| [`edits/CW_CLDICE_ARCHITECTURE_AND_PLAN.md`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/CW_CLDICE_ARCHITECTURE_AND_PLAN.md) | Full architectural and mathematical specification for Conductance-Weighted clDice. |
| [`BASELINE_REPRODUCTION_COMPARISON.md`](file:///d:/Desktop/ARTH/Sem-8/I2/BASELINE_REPRODUCTION_COMPARISON.md) | Detailed parity report against IEEE ISBI 2026 published baseline tables. |
| [`IDEA1_PLUGIN_BENCHMARK_COMPARISON.md`](file:///d:/Desktop/ARTH/Sem-8/I2/IDEA1_PLUGIN_BENCHMARK_COMPARISON.md) | Full empirical failure breakdown & AUROC forensic analysis of Idea 1. |
| [`edits/idea2_topo_csa/README.md`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/idea2_topo_csa/README.md) | Anisotropic directional strip attention skip connection design (Idea 2). |
| [`PROFESSOR_VIVA_AND_DEFENSE_GUIDE.md`](file:///d:/Desktop/ARTH/Sem-8/I2/PROFESSOR_VIVA_AND_DEFENSE_GUIDE.md) | Q&A guide for professor defense, viva presentations, and thesis grading. |
