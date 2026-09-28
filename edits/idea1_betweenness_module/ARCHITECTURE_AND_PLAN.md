# Idea 1: Betweenness-Based Structural Validity Module for SA-UNetv2

## 1. Executive Summary & Core Philosophy

* **Type:** Post-Processing Structural Validation Module (No modification to network architecture, weights, loss, or training loop).
* **Input:** Raw model-predicted probability maps ($P \in [0, 1]$, saved as `float32 .npy`).
* **Core Insight:** Segmentation neural networks frequently produce false-positive offshoots (spurs) or detached micro-islands that look plausible to local convolutional kernels but serve **zero physiological/topological purpose** in the vascular tree.
* **Mechanism:** 
  * Convert vessel prediction into a resistor network (skeleton graph).
  * Assign Poiseuille conductance $c_e = \frac{r_e^4}{L_e + 10^{-3}}$ to each edge based on vessel radius and length.
  * Compute **current-flow betweenness centrality** across node pairs.
  * Real structural vessels bridging the vascular tree experience high electrical current flow ($\text{high } \hat{CB}$). Non-functional dead-end spurs and noise artifacts carry near-zero current ($\text{low } \hat{CB}$).
* **Goal:** Rank, filter, and prune spurious vessels, eliminating fragmentation and boosting topological precision.

---

## 2. Hard Architectural Rules

1. **Model Decoupling:**
   * Run inference once with the reproduced SA-UNetv2 model to dump `float32 .npy` probability maps for all images (`train`, `val`, `test`).
   * The structural module only reads `.npy` files.
2. **Strict Evaluation Integrity (Zero Test-Set Leakage):**
   * Threshold $\tau$ (and any pruning hyperparameters) will be selected exclusively on the validation split (held-out from training).
   * The test set is evaluated exactly once at the final stage with the frozen $\tau$.
3. **Pluggable Scorer Interface:**
   * Function signature: `score_edges(graph, config) -> pd.DataFrame`.
   * Allows plugging in future structural scorers (e.g., Murray's Law branching ratios) without changing the pipeline.
4. **Honest Reporting:**
   * Report all metrics (including where it does not help, e.g. slight drops in recall if spurs are pruned).
5. **Benchmark Parity:**
   * Evaluate before vs. after using our exact global connectivity suite ([`edits/evaluate_connectivity.py`](file:///c:/Users/Student/Arth%20Patel/Btep/edits/evaluate_connectivity.py)) for direct comparability.

---

## 3. Pipeline Stages & Math

### Stage A: Binarize & Clean
* Probability thresholding: $B = (P \ge 0.5)$.
* Apply FOV mask (DRIVE: yes; STARE: none / full frame).
* Remove small isolated noise components ($< 10\text{ px}$).

### Stage B: Skeletonization & Radius Mapping
* Medial axis skeleton: $S = \text{skeletonize}(B)$.
* Distance transform: $R = \text{distance\_transform\_edt}(B)$ (gives local vessel half-thickness / radius at each skeleton pixel).
* Prune skeletonization artifact spurs ($< 5\text{ px}$) before graph construction.

### Stage C: Graph Construction (`sknw` + `networkx`)
* Nodes = Junctions (degree $\ge 3$) and Endpoints (degree $= 1$).
* Edges = Vessel segments connecting nodes, containing ordered coordinate paths `edge['pts']`.
* Segment length: $L_e = \sum \text{Euclidean steps}$ along pixel path (diagonal = $\sqrt{2}$).
* Segment radius: $r_e = \min(R[\text{pts}])$ (bottleneck radius dominating fluid/electrical resistance).

### Stage D: Poiseuille Conductance
* Hydraulic/Electrical conductance:
  $$c_e = \frac{r_e^4}{L_e + 10^{-3}}$$
  *(stored as edge attribute `'conductance'`)*.

### Stage E: Current-Flow Betweenness Centrality
* Computed per connected component of the graph ($G_c$ with $\ge 3$ nodes).
* Exact all-pairs: `nx.edge_current_flow_betweenness_centrality(G_c, weight='conductance')`.
* Optional sampling fallback for large components: solve graph Laplacian $L x = e_s - e_t$.
* Image-level normalization:
  $$\hat{CB}(e) = \frac{CB(e)}{\max_{e'} CB(e')}$$

### Stage F: Outputs
* **Soft Mode:** Keep mask unchanged; render $\hat{CB}$ heatmap overlay + export CSV table with attributes (`id`, `comp`, `length`, `radius`, `conductance`, `CB_hat`, `touches_border`).
* **Hard Mode:** Prune edges where $\hat{CB} < \tau$ (unless touching image/FOV border $\to$ protected). Re-evaluate metrics.

---

## 4. Code Structure (`edits/idea1_betweenness_module/`)

```text
edits/idea1_betweenness_module/
├── io_utils.py          # Save/load prob maps (.npy), ground truth, FOV masks
├── graph_build.py       # Stages A, B, C (binarize, skeleton, distance, sknw graph)
├── scoring.py           # Stages D, E (Poiseuille conductance, current-flow betweenness)
├── apply.py             # Stage F (Soft heatmap overlay, Hard pruning at threshold tau)
├── validate.py          # Edge-level AUROC of (1 - CB_hat) vs FP-segments & tau sweep
├── run_pipeline.py      # Master experiment runner
├── tests/
│   └── test_synthetic.py # Unit tests on synthetic graph (trunk, bridge, loop, spur)
├── checkpoints/         # Model weights (loaded for initial probability map dumping)
├── prob_maps/           # Cached float32 .npy predictions
├── results/             # Stage deliverables (overlays, heatmaps, pruned masks)
└── ARCHITECTURE_AND_PLAN.md # This blueprint
```

---

## 5. Verification Deliverables & Milestones

* [ ] **Milestone 0:** Dump `float32 .npy` probability maps for all DRIVE & STARE images using our reproduced model weights.
* [ ] **Milestone 1 (Stages A–C):** Generate and save **5 visual inspection overlays** (Binary mask, Skeleton, Graph nodes/edges).  
  *(STOP and wait for user inspection/approval before proceeding).*
* [ ] **Milestone 2 (Stages D–E):** 
  - Synthetic graph unit test passing (verifying: Bridge = highest, Spur = near zero, Loop = moderate).
  - Generate and save **5 real eye heatmaps** showing normalized current-flow betweenness $\hat{CB}$.  
  *(STOP and wait for user inspection/approval before proceeding).*
* [ ] **Milestone 3 (Validation & Hard Pruning):**
  - Edge-level AUROC ranking validation against FP ground-truth segments (comparing vs. length alone and radius alone).
  - Validation sweep to select optimal $\tau^*$.
  - Final single evaluation on test set (pixel metrics + connectivity metrics).
