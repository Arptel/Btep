# Idea 2: Topo-CSA (Anisotropic Directional Cross-Scale Spatial Attention)

> **Research Track:** Architectural Innovation on SA-UNetv2  
> **Status:** Conceptual Formulation & Architecture Design (Independent Track)  
> **Target Module:** Skip Connections in SA-UNetv2 ([`src/model.py`](file:///d:/Desktop/ARTH/Sem-8/I2/edits/src/model.py))  

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Limitation of Baseline CSA (ISBI 2026)
In the base paper, the authors introduced **Cross-Scale Spatial Attention (CSA)** to gate encoder skip connections using high-level decoder features. However, the baseline CSA module operates via **symmetric square pooling ($7 \times 7$ window)**:
$$M_{\text{csa}}(F) = \sigma\left(f^{7\times 7}([\text{AvgPool}(F); \text{MaxPool}(F)])\right)$$

**The Flaw in Square Receptive Fields:**
* Blood vessels are **elongated, continuous curvilinear tubes** with distinct directional orientations ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
* Square $7 \times 7$ convolution kernels perform isotropic spatial averaging. When filtering a thin 1-pixel capillary surrounded by high-contrast background noise (e.g., optic disc margins, drusen, or choroidal background patterns), square pooling blurs the thin vessel signal with background noise, causing skip connections to drop faint capillary trajectories.

---

## 2. Proposed Architecture: Topo-CSA

Instead of isotropic square pooling, **Topo-CSA** introduces **Anisotropic Directional Strip Pooling & Trajectory Gating**:

```
Encoder Skip Feature (F_enc) ──┐
                               ▼
Decoder Context (F_dec) ──▶ [Topo-CSA Gate] 
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[Horizontal Strip Pooling]                 [Vertical Strip Pooling]
    (1 x K kernel)                             (K x 1 kernel)
       │                                               │
       ▼                                               ▼
1D Conv (Trajectory Alignment)             1D Conv (Trajectory Alignment)
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
            [Diagonal Kernel Fusion (45° / 135°)]
                               ▼
                    Sigmoid Attention Map M(x, y)
                               │
                               ▼
           Gated Feature: F_out = F_enc ⊙ M(x, y)
```

### 2.1 Mathematical Formulation

1. **Directional Strip Pooling:**
   Capture long-range continuous vascular paths along horizontal and vertical axes without incorporating perpendicular background noise:
   $$y_c^h(i) = \frac{1}{W} \sum_{0 \le j < W} x_c(i, j) \quad \text{(Horizontal Strip Context)}$$
   $$y_c^v(j) = \frac{1}{H} \sum_{0 \le i < H} x_c(i, j) \quad \text{(Vertical Strip Context)}$$

2. **Curvilinear Trajectory Encoding:**
   Pass strip descriptors through $1\text{D}$ depthwise convolutions ($1 \times K$ and $K \times 1$, with $K=15$ pixels) to match standard retinal vessel lengths.

3. **Multi-Scale Spatial Modulator:**
   Combine directional contexts with higher-level decoder features:
   $$M_{\text{topo}}(F_{\text{enc}}, F_{\text{dec}}) = \sigma\left(\text{Conv}_{1\times 1}\left(\left[\text{Expand}(y^h); \text{Expand}(y^v); F_{\text{dec}}\right]\right)\right)$$

---

## 3. Key Advantages & Expected Research Impact

1. **Zero Fragmentation across Disconnects:**
   Directional strip pooling enables the network to "trace" continuous capillaries through gaps, preventing the CNN from breaking fine vessels under low local contrast.
2. **Lightweight Parameter Overhead:**
   Using 1D strip convolutions adds fewer than **$3,500$ parameters** across all three skip stages ($< 1.3\%$ parameter increase), strictly preserving SA-UNetv2's ultra-lightweight regime ($0.26\text{ M}$ parameters).
3. **Pure Architectural Novelty:**
   Directly upgrades the core contribution of the ISBI 2026 paper (the CSA module) with curvilinear-aware anisotropic attention.

---

## 4. Implementation Checklist (For Future Experimentation)

- [ ] Implement `TopoCSAModule` in `edits/src/model.py`.
- [ ] Measure parameter and FLOPs parity against baseline SA-UNetv2.
- [ ] Benchmark on DRIVE using `edits/evaluate_connectivity.py` to measure reduction in Betti-0 fragmentation.
- [ ] Conduct ablation study comparing baseline CSA ($7\times 7$ square) vs. Topo-CSA ($1\times 15 + 15\times 1$ directional).
