# ACE-ProtoNet (MIA 2026): Reference Paper Architecture, Concepts & Research Ideas

> **Status & Role in Project:** **REFERENCE COMPANION PAPER ONLY**  
> *(Our primary Goal #1 is to reproduce the base model **SA-UNetv2** [ISBI 2026]. This document serves as a comprehensive conceptual reference for ideas, techniques, advanced metrics, and future architectural enhancements without needing to reopen the 17-page PDF.)*
>
> **Paper Title:** *ACE-ProtoNet: Adaptive covariance eigen-gate and uncertainty-aware prototype learning for coronary artery segmentation*  
> **Journal:** Elsevier — *Medical Image Analysis* (MIA), Volume 111, 2026, Article 104040  
> **Authors:** Caixia Dong$^{a,b,c}$, Duwei Dai$^{a,b,c}$, Pengyu Ren$^c$, Guowei Dai$^d$, Yu Wang$^e$, Linyun Zhou$^c$, Yang Li$^{f,*}$, Wei Zeng$^{a,b,*}$  
> **Affiliations:** $^a$Xi'an Jiaotong University; $^b$National-Local Joint Engineering Research Center; $^c$Institute of Medical AI; $^d$Sichuan University; $^e$Dept. of Cardiology, 2nd Affiliated Hospital of XJTU; $^f$School of AI, The Chinese University of Hong Kong (Shenzhen)  
> **PDF Filename in Workspace:** `segmentation calibration_MIA2026 (1).pdf`

---

## 1. Executive Summary & Why This Paper is a Valuable Reference

While the base model paper (**SA-UNetv2**) solves the lightweight clinical CPU deployment problem for 2D retinal fundus images, **ACE-ProtoNet** addresses the deep methodological challenges of **complex tubular vascular tree segmentation**:
1. **Severe class imbalance** (vessels occupy $< 2\%$ of the scan volume).
2. **Ambiguous boundaries and low vessel-to-background contrast** in peripheral / distal branches.
3. **Loss of topological continuity** (broken vessels, fragmented capillaries).
4. **Integration of heterogeneous feature streams** (combining global structural semantics with fine-grained local anatomical details).

The paper proposes two major algorithmic contributions:
- **ACE-Gate (Adaptive Covariance Eigen-Gate):** A statistically grounded feature fusion mechanism using **inter-channel covariance analysis** and **eigenvalue decomposition** to adaptively balance features rather than relying on heuristic pooling (like SE-Net) or raw concatenation.
- **UPL-Head (Uncertainty-Aware Prototype Learning Head):** A module that quantifies **voxel-wise predictive uncertainty (via Shannon entropy)**, modulates prototype-guided attention by uncertainty, and updates class prototypes using **uncertainty-weighted soft hard-example mining**.

---

## 2. Core Methodological Components

```
Input Volume
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
[Branch 1: Global VFM]         [Branch 2: Local CNN]         (Skip connections)
SAM-Med3D (ViT, partly frozen)  3D U-Net CNN (Trainable)                 │
   │                              │                                      │
   └──────────────┬───────────────┘                                      │
                  ▼                                                      │
         [ ACE-Gate Module ]                                             │
      (Covariance Matrix Σ + EVD                                         │
        Eigenvalues λ ──> MLP ──> w)                                     │
                  │                                                      │
                  ▼                                                      │
      Fused Features: F_hat = w*F_vfm + (1-w)*F_cnn                      │
                  │                                                      │
                  ▼                                                      │
            [ Decoder ] ◄────────────────────────────────────────────────┘
                  │
                  ├────────────────────────┐
                  ▼                        ▼
           [Auxiliary Head]         [Decoder Features X]
                  │                        │
                  ▼                        │
          Uncertainty Map U                │
          (Shannon Entropy)                │
                  │                        │
                  └──────────────┬─────────┘
                                 ▼
                         [ UPL-Head ]
                  - Cosine Similarity Map S
                  - Prototype Attention X_tilde
                  - Uncertainty Modulation: X_tilde' = X_tilde * (1 + U)
                  - Concat[X_tilde', S] ──> Conv ──> Final Prediction P_f
                                 │
                  (Training: Online EMA Prototype Update with Soft Hard-Mining)
```

---

### 2.1 Adaptive Covariance Eigen-Gate (ACE-Gate)

#### The Problem It Solves
When fusing global representations (e.g., from foundation models or deep bottleneck layers) with local spatial representations (e.g., from shallow CNN layers):
- Naive concatenation or addition leads to feature dominance (one stream overrides the other).
- Squeeze-and-Excitation (SE-Net) or CBAM rely on **first-order statistics** (global average pooling, i.e., mean activations), ignoring channel correlation and variance distributions.

#### Mathematical Formulation

1. **Common Space Alignment:**
   - Dual feature representations: $F_{vfm}, F_{cnn} \in \mathbb{R}^{N \times C_{ch}}$ (where $N = D \cdot H \cdot W$ or $H \cdot W$).
   - Projected into a unified representation $X \in \mathbb{R}^{N \times C_{ch}}$ via $1\times 1$ conv.

2. **Inter-Channel Covariance Matrix:**
   - Mean-center the feature matrix: $G = X - \bar{X}$.
   - Compute the second-order covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{C_{ch} \times C_{ch}}$:
     $$\mathbf{\Sigma} = \frac{1}{N - 1} G^\top G$$
   - This captures pairwise channel co-activation patterns and variance structure across the entire spatial domain.

3. **Eigenvalue Decomposition (EVD):**
   - Solve: $\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$.
   - Eigenvectors $\mathbf{v}_i$ are rotation-variant and sensitive to perturbations, so the paper **discards eigenvectors** and exclusively uses **eigenvalues** $\mathbf{\lambda} = [\lambda_1, \lambda_2, \dots, \lambda_{C_{ch}}]^\top$.
   - Eigenvalues serve as compact, rotation-invariant descriptors of the global variance profile along principal component directions.

4. **Statistical Mapper (Lightweight MLP):**
   - Maps the global variance vector $\mathbf{\lambda}$ into channel-wise gating weights $\mathbf{w}$:
     $$\mathbf{w} = \sigma\big(\text{MLP}(\mathbf{\lambda})\big) \in [0, 1]^{C_{ch} \times 1}$$
   - *Key design principle:* The MLP receives only eigenvalues (not raw spatial activations), enforcing a strict inductive bias based on second-order statistics.

5. **Complementary Soft Gating:**
   $$\hat{F} = \big(F_{vfm} \odot \mathbf{w}\big) + \big(F_{cnn} \odot (1 - \mathbf{w})\big)$$
   - When $w_i \to 1$: Emphasizes global semantic context.
   - When $w_i \to 0$: Emphasizes local high-frequency boundary details.

---

### 2.2 Uncertainty-Aware Prototype Learning Head (UPL-Head)

#### The Problem It Solves
Standard prototype learning creates representative class centroids in latent space. However:
- Normal vessels and background dominate the centroid calculations.
- Faint distal branches, ambiguous bifurcations, and low-contrast boundaries are underrepresented.
- Standard prototypes fail to separate ambiguous vessel pixels from background noise or lesions.

#### Components & Step-by-Step Mechanics

#### 1. Decoupled Uncertainty Estimation
- An auxiliary segmentation head outputs a categorical probability distribution $\mathbf{p}_i = [p_i^{(1)}, \dots, p_i^{(C)}]$ for each voxel/pixel $i$.
- Predictive uncertainty $u_i$ is computed via **Shannon Entropy**:
  $$u_i = -\sum_{c=1}^C p_i^{(c)} \log p_i^{(c)}$$
- The uncertainty map $\mathbf{U} = \{u_i\}$ is normalized to $[0, 1]$ by dividing by its maximum theoretical value, $\log(C)$:
  $$\mathbf{U}_{norm} = \frac{\mathbf{U}}{\log(C)}$$
- High uncertainty directly flags ambiguous, hard-to-classify pixels (e.g., vessel edges, tiny capillaries).

#### 2. Prototype-Guided Prediction & Uncertainty Modulation
Given decoder feature $X \in \mathbb{R}^{N \times C_{ch}}$ and dynamic prototype bank $\{\mu_{c,k}\}$ ($C$ classes, $K$ prototypes per class):
- **Explicit Similarity Map ($S$):** Cosine similarity between voxel feature $x_i$ and each prototype $\mu_{c,k}$:
  $$s_{i,c,k} = \frac{x_i \cdot \mu_{c,k}}{\|x_i\|_2 \|\mu_{c,k}\|_2} \quad \implies \quad S \in \mathbb{R}^{N \times (C \cdot K)}$$
- **Prototype Attention ($\tilde{X}$):** Multi-head cross-attention where voxel features act as Queries ($Q$), and prototypes act as Keys ($K$) and Values ($V$), followed by a residual connection:
  $$\tilde{X} = X + \text{Attention}(Q=X, K=\mu, V=\mu)$$
- **Uncertainty-Guided Modulation:**
  $$\tilde{X}' = \tilde{X} \odot (1 + \mathbf{U})$$
  *Intuition:* Where uncertainty $\mathbf{U}$ is high (hard/ambiguous regions), the prototype context is **upweighted**, compelling the network to rely on robust global class prototypes rather than deceptive local pixel textures.
- **Final Prediction:**
  $$P_f = \sigma\Big(\text{Conv}\big([\tilde{X}' \,;\, S]\big)\Big)$$

#### 3. Online Prototype Learning via Soft Hard-Example Mining
During training, prototypes adapt dynamically:
1. Each pixel/voxel feature $x_i$ is assigned to its nearest intra-class prototype:
   $$k^* = \arg\min_k \|x_i - \mu_{y_i, k}\|_2$$
   forming assignment clusters $\mathcal{I}_{c, k}$.
2. The candidate prototype is updated as an **uncertainty-weighted centroid**:
   $$\mu_{c,k}^{new} = \frac{\sum_{i \in \mathcal{I}_{c,k}} u_i \cdot x_i}{\sum_{i \in \mathcal{I}_{c,k}} u_i}$$
   *Mechanism:* Weighting by $u_i$ performs **soft hard-example mining**. Instead of being swamped by easy background/trunk pixels, the prototype shifts towards challenging, ambiguous vascular patterns.
3. Stable update via **Exponential Moving Average (EMA)**:
   $$\mu_{c,k} \leftarrow (1 - \alpha)\mu_{c,k} + \alpha \mu_{c,k}^{new} \quad (\text{momentum } \alpha = 0.1)$$
   *(During inference, the prototype memory bank is frozen).*

---

### 2.3 Loss Formulations

The overall loss balances segmentation accuracy with metric learning:
$$\mathcal{L}_{total} = \mathcal{L}_{seg} + \beta \mathcal{L}_{ppc} \quad (\beta = 0.2)$$

1. **Hybrid Segmentation Loss ($\mathcal{L}_{seg}$):**
   $$\mathcal{L}_{seg} = \mathcal{L}_{WCE} + \mathcal{L}_{DSC}$$
   - $\mathcal{L}_{WCE}$: Weighted Cross-Entropy to handle foreground-background imbalance.
   - $\mathcal{L}_{DSC}$: Soft Dice loss to maximize overlap and preserve spatial continuity.

2. **Voxel-Prototype Contrastive Loss ($\mathcal{L}_{ppc}$):**
   Pulls each pixel feature towards its assigned positive prototype $\mu_{c^+, k^+}$ while pushing it away from all negative prototypes across the memory bank $\mathcal{M}$:
   $$P(\mu_{c^+, k^+} \mid x_i) = \frac{\exp\big(\text{sim}(x_i, \mu_{c^+, k^+}) / \tau\big)}{\sum_{\mu \in \mathcal{M}} \exp\big(\text{sim}(x_i, \mu) / \tau\big)}$$
   $$\mathcal{L}_{ppc} = -\frac{1}{|\mathcal{V}_{label}|} \sum_{i \in \mathcal{V}_{label}} \log P(\mu_{c^+, k^+} \mid x_i)$$
   where $\tau$ is a temperature hyperparameter. Enforces high intra-class compactness and inter-class separability.

---

## 3. Key Evaluation Metrics Introduced in the Reference Paper

In addition to standard metrics (DSC/F1, Sensitivity, Specificity, Accuracy), the paper emphasizes two advanced metrics crucial for vascular structures:

### 3.1 Centerline Dice (`clDice`) — Topology & Connectivity
Standard Dice/F1 measures volume/area overlap, which means a model can sever a thin vessel branch but still achieve a 95% Dice score.
- `clDice` extracts the topological skeleton (centerline) $S_P = \text{skeletonize}(P)$ and $S_G = \text{skeletonize}(G)$:
  $$\text{Tprec} = \frac{|P \cap S_G|}{|S_G|}, \quad \text{Tsens} = \frac{|S_P \cap G|}{|S_P|}$$
  $$\text{clDice} = \frac{2 \cdot \text{Tprec} \cdot \text{Tsens}}{\text{Tprec} + \text{Tsens}}$$
- Directly penalizes topological breakages, gaps, and disconnected branches.

### 3.2 Average Symmetric Surface Distance (`ASSD`) — Boundary Precision
Measures the average Euclidean distance between the predicted boundary surface voxels/pixels $P_S$ and ground-truth boundary $G_S$:
$$\text{ASSD} = \frac{1}{|P_S| + |G_S|} \left( \sum_{p \in P_S} \min_{g \in G_S} d(p, g) + \sum_{g \in G_S} \min_{p \in P_S} d(g, p) \right)$$
- Lower ASSD indicates accurate capillary edge delineation without fuzzy dilation or erosion.

---

## 4. Key Experimental & Ablation Takeaways

From the extensive ablation studies reported on CTA119, ASOCA, and ImageCAS:

1. **Second-Order Covariance vs. First-Order Pooling (Table 4 in paper):**
   - Replacing basic addition or concatenation with ACE-Gate increases Dice by **$+1.8\%$ to $+2.4\%$** across datasets.
   - Proves that eigenvalue variance profiles provide cleaner cross-stream gating than channel average pooling.
2. **Number of Prototypes $K$ (Table 6 in paper):**
   - Testing $K \in \{1, 2, 3, 5\}$:
   - $K=1$: Dice $86.15\%$, clDice $91.45\%$
   - $K=2$: Dice $86.92\%$, clDice $92.10\%$
   - **$K=3$ (Optimal):** **Dice $87.34\%$**, **clDice $92.79\%$**
   - $K=5$: Dice drops slightly to $87.09\%$ (too many prototypes over-fragments the embedding space).
   - *Takeaway:* Having 2–3 sub-prototypes per class (e.g., vessel trunk vs. thin capillary vs. branching node) is ideal.
3. **Uncertainty-Weighted Mining:**
   - Weighting prototype updates by Shannon entropy $u_i$ prevents prototypes from collapsing into majority background features, maintaining sensitivity to fine distal vessels.

---

## 5. Actionable Roadmap & Ideas to Borrow for Our Retinal Project

> **Reminder:** We must keep our immediate focus on **Phase 1: Reproducing the base SA-UNetv2 model**.  
> The ideas below represent high-value concepts from ACE-ProtoNet to explore in subsequent research phases.

| Idea from ACE-ProtoNet | Potential Application in Retinal Vessel Project | Expected Benefit / Research Value |
| :--- | :--- | :--- |
| **Idea 1: Topology-Aware Metric (`clDice`)** | Add `clDice` calculation using `skimage.morphology.skeletonize` to our evaluation pipeline on DRIVE and STARE. | Gives a definitive quantitative measure of whether thin capillaries are connected or broken, complementing F1 and Jaccard. |
| **Idea 2: Predictive Uncertainty Map via Shannon Entropy** | Compute $u_i = -[p \log p + (1-p)\log(1-p)]$ at test time for fundus images. | Identifies ambiguous vessel boundaries and optic disc borders; produces clinically useful confidence heatmaps alongside binary masks. |
| **Idea 3: Uncertainty-Modulated Loss / Hard Mining** | Weight the MCC or BCE loss dynamically by pixel entropy $(1 + u_i)$ during training. | Forces the model to focus gradient updates on faint, low-contrast capillaries without increasing parameter count. |
| **Idea 4: Multi-Prototype Representation for Retinal Features** | Maintain $K=2$ or $3$ prototypes for Foreground (e.g., thick primary vessels vs. thin capillaries) and Background (normal retina vs. optic disc vs. lesions). | Reduces false positives caused by pathological exudates, drusen, or bright optic cup margins misidentified as vessels. |
| **Idea 5: Lightweight Covariance Skip-Gating** | Upgrade SA-UNetv2's CSA module from channel average pooling to a mini-covariance eigenvalue gate at the skip connections. | Enhances cross-scale semantic fusion while maintaining a sub-million parameter budget. |

---

## 6. Summary Comparison: Base Paper vs. Reference Paper

| Feature | Base Paper: SA-UNetv2 (ISBI 2026) | Reference Paper: ACE-ProtoNet (MIA 2026) |
| :--- | :--- | :--- |
| **Target Task** | 2D Retinal Vessel Segmentation (DRIVE, STARE) | 3D Coronary Artery Segmentation (CCTA: ASOCA, ImageCAS, etc.) |
| **Primary Project Role** | **Our Base Model to Reproduce (Goal #1)** | **Reference for Conceptual Ideas & Extensions** |
| **Model Size / Focus** | Ultra-lightweight ($0.26\text{M}$ params, $1.2\text{MB}$, sub-second CPU inference) | Large hybrid framework (SAM-Med3D Foundation Model + 3D CNN) |
| **Key Architecture Trick**| Cross-scale Spatial Attention (CSA) on skips + GroupNorm + SiLU | ACE-Gate (Covariance EVD) + UPL-Head (Uncertainty Prototypes) |
| **Class Imbalance Strategy**| Compound Loss: $0.5 \times \text{BCE} + 0.5 \times \text{MCC}$ (continuous MCC) | Weighted CE + Dice + Voxel-Prototype Contrastive Loss ($\mathcal{L}_{ppc}$) |
| **Uncertainty Awareness** | Implicit via MCC correlation | Explicit via Shannon entropy map $U$, modulation, and soft hard-mining |
| **Key Takeaway for Us** | Exact architecture and pipeline to implement and train now | Reservoir of advanced loss, calibration, and gating ideas for novel extensions |
