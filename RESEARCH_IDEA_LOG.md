# Research Idea Log - Retinal Vessel Segmentation

## Purpose

This is a **living brainstorming and decision log**, not a list of claimed novel contributions. An idea may be scientifically interesting and still be unsuitable because it is already published, cannot be evaluated fairly with the available data, or is too large for this project.

Use this record before starting an experiment. For every candidate, answer:

1. What specific baseline failure does it address?
2. What is the proposed mechanism?
3. What existing work is closest to it?
4. What is genuinely different?
5. What result would count as success or failure?

Current baseline: **SA-UNetv2**, a lightweight U-Net with Cross-scale Spatial Attention (CSA) and a BCE + continuous-MCC loss. The immediate baseline task is complete on DRIVE and remains to be reproduced on STARE.

## Status legend

| Label | Meaning |
|---|---|
| `ACTIVE CANDIDATE` | Worth a focused novelty audit and small feasibility experiment. |
| `BACKGROUND / ABLATION` | Useful idea or diagnostic, but not a stand-alone contribution. |
| `DO NOT CLAIM AS NOVEL` | Directly covered, or too close to existing work. |
| `NOT YET ASSESSED` | A hypothesis retained only for later evaluation. |

## Project-level problem statement

The practical difficulty is not simply producing a connected binary mask. SA-UNetv2 can still miss faint/thin vessels, confuse background texture with vessels, and behave differently across cameras or datasets. A good extension must preserve the baseline's lightweight nature, keep ordinary segmentation quality close to or above the baseline, and make a meaningful improvement in at least one of:

- thin-vessel recall;
- segmentation reliability or calibration;
- cross-dataset robustness;
- anatomical/topological validity;
- inference efficiency or memory.

## Candidate 1 - Agreement-Aware SA-UNetv2

**Status:** `ACTIVE CANDIDATE - novelty audit required`

### What

Extend SA-UNetv2 into a small multi-task model that predicts both:

1. a normal vessel probability map; and
2. an ambiguity/reliability map estimating where trained human annotators are likely to disagree.

The core claim would be that a segmentation system should distinguish clear vessels from genuinely ambiguous, faint, boundary, or thin-vessel regions instead of outputting one equally confident binary decision everywhere.

### Motivation

Retinal-vessel masks are not perfect physical ground truth. Thin vessels and boundaries are difficult to label consistently. Standard BCE, Dice, MCC, and topology losses treat every annotated pixel as equally certain. A model may therefore be confidently wrong on the very locations where experts themselves disagree.

This direction aims to make the output more clinically honest: preserve the vessel mask, while identifying regions where it should be reviewed rather than trusted blindly.

### Proposed supervision

For images that have two independent masks, let `Y1` and `Y2` be binary masks.

```text
Consensus target:     Y = (Y1 + Y2) / 2
Disagreement target:  D = |Y1 - Y2|
```

- `Y = 1`: both observers say vessel.
- `Y = 0`: both observers say background.
- `Y = 0.5`: observers disagree.
- `D = 1`: disagreement; `D = 0`: agreement.

The model has a segmentation head `P` and a reliability head `R`.

```text
L_total = L_segmentation(P, Y) + lambda_agree * L_reliability(R, D)
```

Initially, `L_segmentation` should retain the baseline BCE + continuous MCC design. The reliability head should be a minimal decoder-side head so that parameter and inference-cost changes remain negligible.

### Rough implementation outline

1. Verify which datasets contain two masks per image and their licensing/splits.
2. Ensure that all annotations of one image stay in the same split. Never train on one observer's annotation and test on another annotation of the same image.
3. Add a one-channel reliability head beside the existing final segmentation head.
4. Train a plain multi-task version before attempting any reliability-guided segmentation refinement.
5. Evaluate ordinary F1/Jaccard/MCC plus:
   - ability of `R` to predict observer disagreement;
   - calibration metrics such as Brier score / expected calibration error where appropriate;
   - risk-versus-coverage: if the model abstains on its most ambiguous pixels, does error fall on the remaining pixels?
6. Compare against probability entropy as the simple non-learned uncertainty baseline.

### Important experimental rules

- DRIVE's second observer mask, if used, must be held out from training when it belongs to the official test images.
- STARE's two observer masks should be checked carefully; use image-level splits, not annotation-level splits.
- A larger multi-annotator dataset such as FIVES may be needed for a convincing training and evaluation protocol.
- The primary vessel mask must still be evaluated at the conventional threshold and against standard masks. Reliability must not become a way to hide poor segmentation quality by abstaining everywhere.

### Closest known work / novelty risk

- Retinal-vessel literature recognizes inter-observer variability and the limitations of pixel-only evaluation. The RETA benchmark discusses these issues and topology-aware evaluation.
- Generic uncertainty estimation, uncertainty-aware labels, and uncertainty-guided refinement already exist. Do **not** claim that uncertainty itself is new.
- The exact claim to verify is narrower: whether retinal-vessel segmentation has already jointly learned a disagreement-prediction head from multiple human annotations and evaluated disagreement-calibrated selective predictions across datasets.

### What would make this publishable or not

Promising result:

- baseline-level or better segmentation F1;
- reliability map significantly predicts actual observer disagreement better than entropy alone;
- selective high-confidence predictions have demonstrably lower error;
- results hold on a held-out dataset or a rigorous image-level split.

Failure conditions:

- the reliability map merely copies `0.5 - |P - 0.5|` and provides no information beyond probability entropy;
- segmentation quality drops materially;
- there are insufficient independent annotations for a leakage-free evaluation.

## Candidate 2 - Murray's-Law-Informed Bifurcation Loss

**Status:** `DO NOT CLAIM AS NOVEL`

### What

The proposal was to regularize segmentation using a branching-width relation inspired by Murray's law:

```text
r_parent^alpha approximately equals r_daughter_1^alpha + r_daughter_2^alpha
```

The initial version used `alpha = 3`, radius estimates from a distance transform, soft skeletonization to identify bifurcations, and a training-only auxiliary loss. This is scientifically motivated because vascular branching is a transport-network process rather than merely a generic connected-tube geometry.

### Motivation

Topology losses can reward connectivity even when a predicted branch has implausible geometry. A branching-width regularizer could, in principle, penalize anatomically implausible Y junctions, severe radius jumps, or some false branches without increasing inference cost.

### Why it is not the main contribution

In 2026, the preprint **MARVEL: Universal Murray's Law-informed Vessel Tree Segmentation and Topology Estimation** directly introduces Murray-law-informed vascular segmentation with explicit radius prediction and differentiable bifurcation regularizers. It is too close to claim this direction as original, even if MARVEL has not yet completed peer review.

### Scientific and implementation cautions

- Ordinary SciPy distance transforms and conventional skeletonization are not differentiable. They cannot be placed in an end-to-end loss without a differentiable approximation.
- A missing branch can eliminate a detectable bifurcation entirely; therefore, a bifurcation-only loss cannot necessarily penalize every disconnection.
- A false branch can accidentally satisfy the relation.
- Fundus masks measure apparent 2D width, not true 3D lumen radius. Width depends on resolution, camera scale, annotation style, and preprocessing.
- Retinal bifurcations do not universally follow the cubic law. Reported optimal exponents vary by artery/vein, calibre, and disease state.
- A hard physiological prior can suppress clinically meaningful abnormal morphology.

### Retained use

Use Murray-style measures only as a possible **background diagnostic** or descriptive anatomical analysis if it contributes to error interpretation. Do not implement it as the headline method or represent it as a new loss.

## Candidate 3 - Structural Mask Teacher / Learned Shape Prior

**Status:** `DO NOT CLAIM AS NOVEL`

### What

Train a separate encoder or autoencoder on vessel masks, then use its learned representation to guide image-to-mask segmentation.

### Motivation

The mask encoder learns statistical regularities of vessel trees and could penalize implausible structural output.

### Existing work / decision

This is very close to Yuan et al., *Improving Vessel Connectivity in Retinal Vessel Segmentation via Adversarial Learning* (Knowledge-Based Systems, DOI: 10.1016/j.knosys.2022.110243). That work uses an autoencoder-derived vessel structural prior and adversarial learning to improve connectivity. It should be cited as related work, not rebuilt as the project's novelty.

## Candidate 4 - Topology Loss, clDice, Skeleton Loss, GNN Refinement

**Status:** `DO NOT CLAIM AS NOVEL`

### What

Combine differentiable skeleton/topology losses with graph reasoning or a graph neural network to repair discontinuous vessels.

### Motivation

Pixel metrics can miss broken vessels, poor branches, and disconnected components.

### Existing work / decision

Topology-aware losses, connection-sensitive segmentation, graph reasoning, and GNN refinement are already an active retinal-vessel research area. A generic combination would be incremental and difficult to defend as novel. It may still be included as an evaluation baseline if required.

## Candidate 5 - Strip / Sequence / Directional Continuation Module

**Status:** `DO NOT CLAIM AS NOVEL`

### What

Use horizontal/vertical strips, directional convolutions, sequential attention, or continuation prediction to follow vessel paths.

### Motivation

Vessels are elongated, directional objects. Square convolution kernels can fail to model long curvilinear continuity and branch orientation.

### Existing work / decision

Directional and strip-based retinal-vessel feature extraction already exists, including RVS-FDSC and orientation-aware networks such as OCE-Net. A simple directional/strip module is therefore not a sufficiently distinctive central contribution.

## Candidate 6 - Generic Domain Adaptation / Teacher-Student Pseudo-Labels

**Status:** `DO NOT CLAIM AS NOVEL`

### What

Train on one dataset and adapt to another without target labels using teacher-student pseudo-labels, adversarial alignment, or feature normalization.

### Motivation

Fundus datasets differ in camera, illumination, colour response, scale, contrast, and pathology; models can lose performance under such domain shift.

### Existing work / decision

This is a valid and important problem but already has direct DRIVE-to-STARE and STARE-to-DRIVE solutions. It is too large and crowded for the current project unless a sharply different, well-verified mechanism emerges.

## Candidate 7 - Generic Entropy-Uncertainty Iterative Refinement

**Status:** `DO NOT CLAIM AS NOVEL`

### What

Estimate entropy of prediction, attend to high-entropy pixels, and iteratively refine the vessel mask.

### Existing work / decision

This has direct recent precedent in uncertainty-guided iterative retinal-vessel refinement. It is not a viable headline contribution. It remains useful as a baseline comparison for Candidate 1, where ordinary entropy is compared against disagreement-supervised reliability.

## Candidate 8 - Physiology/Anatomy-Informed Reliability Flagging

**Status:** `NOT YET ASSESSED`

### What

Use anatomical diagnostics such as unexpected branch-width relationships, implausible crossings, or unusual vessel geometry to flag a prediction as potentially unreliable, without forcing the segmentation to obey a hard physiological rule.

### Motivation

This preserves the useful clinical insight in physiology-informed reasoning while avoiding the dangerous assumption that every pathological vessel must conform to a healthy optimality law.

### Relationship to Candidate 1

It could later become an auxiliary input to the reliability head. It must not be added until Candidate 1 is established, and it needs a separate literature review because MARVEL already covers closely related Murray-law-informed topology estimation.

## Candidate 9 - OpenAI/API/Vision-Model Segmentation

**Status:** `BACKGROUND ONLY - not a project model`

External vision APIs may be explored for a qualitative comparison or an idea-generation tool, but they are not a reproducible, locally trained, academically defensible replacement for the project model. They must not be presented as the project's architecture or used to generate hidden test predictions.

## Candidate 10 - Kirchhoff / Poiseuille Flow-Plausibility Graph Analysis

**Status:** `BACKGROUND / EXPLORATORY - not automatic pruning`

### What

Convert a predicted vessel mask into a centerline graph. Estimate each segment's apparent radius from the mask and assign a Poiseuille-inspired conductance:

```text
conductance approximately proportional to radius^4 / segment_length
```

Then solve a graph-flow or Kirchhoff-style system under chosen boundary conditions and use the resulting segment score to identify anatomically suspicious parts of the predicted network.

### Motivation

Connectivity alone is not a sufficient definition of an anatomically credible vessel tree. A segmentation may be connected but contain a highly implausible narrow bridge, radius discontinuity, or short spurious spur. Global graph reasoning could provide a continuous plausibility score rather than a simple connected-component filter.

### Proposed exploratory pipeline

1. Obtain a binary or probability vessel mask from SA-UNetv2.
2. Extract a skeleton graph: junctions/endpoints become nodes and centerline segments become edges.
3. Estimate apparent segment radius from a distance transform or local mask width.
4. Assign edge resistance using a Poiseuille-inspired relation such as `length / radius^4`.
5. Choose explicit boundary conditions and solve the graph system.
6. Visualize low-support / low-plausibility segments as **review flags**, not deletion targets.
7. Compare flagged regions against false positives, false negatives, annotation disagreement, and manual visual review.

### Why it is not presently an automatic correction method

The simple claim that the optic disc is one source and every endpoint is a sink is biologically incorrect for a combined binary fundus mask. Retinal arteries flow outward from the disc and veins flow back toward it; a valid flow model needs artery/vein separation plus justified pressure or flow boundaries. A real thin capillary can carry low flow, while a false endpoint marked as a sink can receive modelled current. Therefore low simulated current is not ground truth evidence that a segment should be removed.

Additional risks:

- apparent 2D mask width is not true 3D lumen radius and varies with camera scale, resolution, preprocessing, and annotation style;
- visible endpoints may be genuine terminals, cropped vessels, or missing annotations;
- missing predicted branches may not exist in the graph and cannot be recovered by graph pruning;
- threshold selection still requires labelled validation data;
- skeletonization, graph construction, optic-disc/artery-vein inference, and flow solving have real inference cost and must be benchmarked;
- disease-related morphology may be clinically meaningful rather than an error to prune.

### Existing work / novelty risk

Physics-informed retinal vascular modelling and Poiseuille-flow simulation already exist. This exact use as post-segmentation Kirchhoff-style plausibility analysis may warrant a later narrow literature audit, but it must not be claimed as novel today. It also overlaps conceptually with physics-informed vascular segmentation and topology-estimation work such as MARVEL.

### Potential useful role

The safest use is as an auxiliary **anatomy-inspired diagnostic** alongside Candidate 1 (Agreement-Aware SA-UNetv2). The system can show a segmentation probability, a learned disagreement/reliability map, and a graph-plausibility flag. It must not automatically erase low-score vessels unless a rigorous, label-based experiment demonstrates that this helps more than it harms.

## Sources to verify before citing

Only cite primary or authoritative sources in final work: publisher pages with DOI, PubMed/PMC records, IEEE Xplore, official conference proceedings, official dataset papers, and original author repositories. Do not treat an AI summary, ResearchGate page, blog, or search-result excerpt as novelty evidence.

Key starting references:

- SA-UNetv2 base paper and official repository.
- Yuan et al., 2023, structural-prior connectivity method, DOI: 10.1016/j.knosys.2022.110243.
- AADG, IEEE TMI 2022, domain generalization, DOI: 10.1109/TMI.2022.3193146.
- Yue et al., 2025, DRIVE/STARE unsupervised domain adaptation, DOI: 10.1038/s41598-024-83018-x.
- FIVES dataset, Scientific Data 2022, DOI: 10.1038/s41597-022-01564-3.
- MARVEL, 2026 preprint: direct Murray-law-informed segmentation precedent. Treat as novelty-blocking prior art until its publication status changes.

## Decision gate before implementation

Do not implement a new research module until all boxes are answered in a short experiment note:

- [ ] Baseline reproduced on DRIVE and STARE.
- [ ] Exact task failure is shown with examples and metrics.
- [ ] Closest prior work is identified from a credible source.
- [ ] Difference from prior work is one sentence and technically testable.
- [ ] Dataset split does not leak labels or annotations.
- [ ] Minimal ablation plan is specified.
- [ ] Success metric includes standard segmentation quality and the claimed additional benefit.
