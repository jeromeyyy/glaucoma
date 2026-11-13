# GLUCOM Autonomous Engineering Mission: Architecture & Repository Audit Report

**Date:** 2026-11-13  
**Engine:** Gemini 3.8 Flash  
**Status:** Stage 1 Repository Audit Complete  
**Git Checkpoint:** `58e321e` (`checkpoint(stage-1): initial backup and repository audit of exported Glucom codebase`)

---

## 1. Executive Summary

An exhaustive repository inspection and audit was performed on the Glucom codebase exported from Emergent AI. The application implements an advanced Explainable AI (XAI) clinical glaucoma detection platform titled **OCULAR / XG: Explainable Glaucoma Detection Using EfficientNet-B3, Hypergraph Neural Networks, and Vision-Language Transformers**.

The architecture combines convolutional deep feature extraction (EfficientNet-B3), spatial-semantic-vascular hypergraph modeling ($N=64$ nodes, $E=40$ hyperedges), differentiable sub-hypergraph selection via Gumbel-Softmax, model-derived clinical concept extraction, and cross-attention Vision-Language Transformer (VLT) narrative generation.

---

## 2. Component-by-Component Repository Audit

### 2.1 Frontend Architecture
- **Framework:** Jinja2 server-rendered templates (`templates/base.html`, `templates/index.html`, `templates/result.html`, `templates/ablation.html`) powered by modern vanilla JavaScript (`static/js/app.js`) and high-fidelity cyber-medical responsive CSS (`static/css/styles.css`).
- **User Interface Features:**
  - Ambient glow dark research interface with glowing green/emerald accents.
  - Interactive drag-and-drop retinal fundus upload zone with instant preview.
  - One-click sample image loaders for Normal and Glaucoma cases.
  - Multi-modal inference results display:
    - Diagnostic verdict with confidence percentage badge.
    - CLAHE-enhanced fundus view.
    - Grad-CAM saliency overlay.
    - X-HGNN sub-hypergraph connectivity topology overlaid on fundus anatomy.
    - Model-derived clinical concept radar chart.
    - Vision-Language Transformer (VLT) structured clinical narrative.
    - Direct PDF Clinical Report download.
  - Baseline comparison & ablation study dashboard (`/ablation`).
- **Preservation Mandate:** 100% of the UI layout, styling, interaction flows, visual widgets, and branding are preserved.

### 2.2 Backend Architecture & API Contracts
- **Original Engine:** Flask 3.0+ in `app.py`.
- **Target Production Backend:** Production FastAPI backend with complete REST API contract:
  - `GET /health` — Service health, device info, checkpoint status.
  - `GET /model/info` — Model architecture, parameters, hyperedge breakdown, dataset provenance.
  - `POST /predict` — Secure multipart image upload, CLAHE enhancement, real model inference.
  - `POST /explain` — Dedicated explainability service (Grad-CAM, X-HGNN sub-hypergraph, radar chart, VLT).
  - `GET /history` and `GET /history/{case_id}` — Persistent case history.
  - `GET /download_pdf/{case_id}` — Multi-modal PDF clinical report export via ReportLab.
  - `GET /api/sample/{sample_type}` — Sample fundus loading for one-click testing.
  - `GET /ablation` and `GET /api/comparison` — Quantitative baseline benchmark results.
- **Data Persistence:** Replaces ephemeral in-memory cache with an automated persistent SQLite/SQLAlchemy database to ensure prediction records and clinical metadata survive restarts.

### 2.3 Model & HGNN Architecture
- **Backbone:** PyTorch `EfficientNet-B3` pretrained on ImageNet (cached locally on machine: `efficientnet_b3_rwightman-b3899882.pth`), extracting intermediate feature maps $\mathbf{F} \in \mathbb{R}^{B \times 1536 \times 10 \times 10}$.
- **Localized Node Extractor:** 8x8 adaptive pooling yielding $N = 64$ localized patch nodes with normalized 2D spatial coordinates $[-1, 1]$, projected to 256-dimensional node embeddings.
- **Hypergraph Construction ($E = 40$ hyperedges):**
  - **16 Spatial Hyperedges:** 4 quadrants (ST, SN, IT, IN), 4 concentric rings (optic disc center, peripapillary, mid-periphery, outer), and 8 octants (ISNT clinical sectors).
  - **16 Semantic Hyperedges:** Dynamic $k$-NN feature clustering in deep representation space ($k=6$).
  - **8 Vascular & Neuroretinal Rim Hyperedges:** Superior/inferior temporal vascular arcades, horizontal/vertical rim contours, central cup-disc boundary, and peripapillary crescent zones.
- **Explainable HGNN Convolution (X-HGNN):** 2-stage hypergraph convolution:
  $$\mathbf{Z} = \sigma\left(\mathbf{D}_v^{-1/2} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^T \mathbf{D}_v^{-1/2} \mathbf{X} \mathbf{\Theta}\right)$$
- **Gumbel-Softmax Differentiable Sub-Hypergraph Selector:** Learns discrete selection weights over the 40 hyperedges with temperature annealing, selecting top-$k=12$ decision-critical hyperedges with sparsity and entropy regularization losses.
- **Clinical Concept Extractor:** Evaluates 5 model-derived clinical concepts:
  1. Optic Disc Morphology
  2. Optic Cup Excavation
  3. Cup-to-Disc Ratio Proxy (CDR-P)
  4. Neuroretinal Rim Thinning Risk
  5. Retinal Vessel Attenuation & Nasal Shifting
- **Vision-Language Transformer (VLT):** Cross-attention Transformer decoder fusing visual node tokens, clinical concept tokens, and diagnostic status to generate grounded clinical reasoning narratives.

### 2.4 Preprocessing & Real Datasets
- **Preprocessing:** CLAHE enhancement on the green channel ($G \times 0.6 + R \times 0.2 + B \times 0.2$), bilateral filtering, image resizing to $300 \times 300$, and standard ImageNet normalization.
- **Identified Real Datasets (Located on Machine in `Downloads/archive`):**
  - **ACRIMA:** 705 real clinical fundus images (396 glaucomatous, 309 normal), gold-standard expert annotations.
  - **Fundus_Train_Val_Data:** 1,306 clinical fundus images structured into `Train` and `Validation` folders for `Glaucoma_Positive` and `Glaucoma_Negative`.
  - **ORIGA:** 650 clinical images with `glaucoma.csv` providing ground-truth Cup-to-Disc Ratio (`ExpCDR`) annotations.
- **Zero Patient/Image Leakage:** Partitioning strictly isolates train, validation, and test splits with deterministic seeding.

