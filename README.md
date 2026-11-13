# Explainable Glaucoma Detection Using EfficientNet-B3, Hypergraph Neural Networks, and Vision-Language Transformers

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Research Overview & Extension Context

This final-year research project is an **extension of an earlier mini-project**:
> *"Glaucoma Detection Using Deep Learning on Retinal Fundus Images"* (which used a standard 4-block custom CNN feeding into a simple Flask binary classifier).

While the original mini-project operated strictly as a black-box binary classifier, this final-year research framework develops a comprehensive, multi-modal **Explainable AI (XAI)** clinical screening system:

```
[Retinal Fundus Image]
        ↓
[CLAHE Preprocessing & Normalization]
        ↓
[EfficientNet-B3 Backbone (Intermediate 2D Feature Map)]
        ↓
[Localized Retinal Feature Extraction (8x8 Grid → N=64 Nodes + 2D Spatial Coords)]
        ↓
[Hypergraph Construction (Spatial, Semantic k-NN, Vascular Arcades: E=40)]
        ↓
[Explainable Hypergraph Neural Network (X-HGNN Convolution)]
        ↓
[Gumbel-Softmax Differentiable Sub-Hypergraph Selector]
        ↓
[Model-Derived Clinical Concepts (CDR Proxy, Rim Integrity, Cup Excavation, Disc Margin, Vessels)]
        ↓
[Unified Glaucoma Classifier (Normal vs. Glaucoma + Confidence %)]
        ↓
    ┌───────────────────────────┴───────────────────────────┐
    ↓                                                       ↓
[Grad-CAM Visual Heatmap & Overlay]       [Vision-Language Transformer (VLT)]
    └───────────────────────────┬───────────────────────────┘
                                ↓
        [Unified Multimodal Clinical Report (Web UI & PDF Export)]
```

---

## ⚡ Quick One-Click Startup (Single Command)

To verify dependencies, set up sample data, verify checkpoints, launch the server, and automatically open the web dashboard in your browser, run **any ONE** of the following:

### Option A: Standard Terminal / PowerShell
```powershell
python run.py
```

### Option B: Windows Batch (or Double-Click in File Explorer)
```cmd
run.bat
```

### Option C: PowerShell Script
```powershell
.\run.ps1
```

The script automatically:
1. Checks and verifies all required Python libraries.
2. Ensures the sample fundus dataset is initialized.
3. Confirms model checkpoints exist (or runs an initial calibration).
4. Starts the Flask application server on `http://127.0.0.1:5000`.
5. Automatically opens `http://127.0.0.1:5000` in your default browser.

---

## 🔬 Core Novelty & Methodological Contributions

1. **Backbone Upgrade**: Replaced the custom 4-layer CNN with **EfficientNet-B3**, exposing deep intermediate convolutional feature maps ($1536 \times 10 \times 10$).
2. **Beyond Pairwise Graphs**: Rather than ordinary Graph Neural Networks (GNNs) connecting pairs of pixels, this architecture constructs a **Hypergraph** where each hyperedge connects **multiple nodes simultaneously** to capture higher-order retinal relationships:
   - **16 Spatial Hyperedges**: Quadrants, concentric retinal zones (optic cup center, peripapillary zone, mid-periphery, outer periphery), and ISNT sectors.
   - **16 Semantic Feature Hyperedges**: k-NN clusters in deep feature space connecting regions with similar pathological features (e.g. pallor, atrophy) across distant locations.
   - **8 Vascular & Neuroretinal Rim Hyperedges**: Superior/inferior temporal vascular arcades and disc boundary contours.
3. **End-to-End Differentiable Selection**: Implemented **Gumbel-Softmax sampling** during training to learn a compact, decision-relevant sub-hypergraph $\mathbf{H}_{sub}$ without ad-hoc thresholding.
4. **Model-Derived Clinical Concepts**: Derives 5 quantitative visual concepts from the selected sub-hypergraph:
   - Optic Disc Morphology
   - Optic Cup Characteristics
   - Cup-to-Disc Ratio Proxy (CDR-P)
   - Neuroretinal Rim Integrity
   - Retinal Vessel Attenuation & Nasal Shifting
5. **Vision-Language Transformer (VLT)**: A cross-attention transformer module that attends across visual features and concept representations to generate natural-language clinical narratives.
6. **Unified Clinical Report Generator**: Real-time generation of multi-modal web reports and formal downloadable PDF clinical summaries via ReportLab.

---

## 📂 Project Structure

```
glaucoma_xhgnn_vlt/
├── app.py                          # Flask backend with Web Dashboard and REST API
├── config.py                       # Global hyperparameters, device configuration, paths
├── requirements.txt                # Python package dependencies
├── README.md                       # Complete research documentation and guide
│
├── models/
│   ├── baseline_cnn.py             # Mini-project baseline 4-block CNN
│   ├── efficientnet_b3.py          # Pretrained EfficientNet-B3 backbone
│   ├── hypergraph.py               # Retinal patch extraction & hypergraph constructor
│   ├── x_hgnn.py                   # Hypergraph Convolution (HGNN) layers
│   ├── gumbel_selector.py          # Gumbel-Softmax differentiable sub-hypergraph selector
│   ├── clinical_extractor.py       # Model-derived clinical concept probes
│   ├── vlt.py                      # Vision-Language Transformer explanation generator
│   └── full_model.py               # Unified end-to-end X-HGNN-VLT model & ablations
│
├── preprocessing/
│   ├── transforms.py               # CLAHE retinal contrast enhancement & transforms
│   ├── augmentation.py             # Albumentations training & validation pipelines
│   └── preprocessing.py            # Image loader, format validation & tensor formatting
│
├── explainability/
│   ├── gradcam.py                  # Gradient-weighted Class Activation Mapping (Grad-CAM)
│   ├── hypergraph_explainer.py     # Sub-hypergraph spatial grounding on fundus image
│   ├── clinical_concepts.py        # Concept card formatter & 5-axis radar chart
│   └── vlt_explainer.py            # High-level VLT clinical narrative service
│
├── training/
│   ├── train.py                    # Multi-objective end-to-end training pipeline
│   ├── validate.py                 # Evaluation loop on validation/test split
│   ├── losses.py                   # Joint classification + sparsity + entropy loss
│   └── checkpoints.py              # Checkpoint saver/loader with metadata
│
├── evaluation/
│   ├── metrics.py                  # Accuracy, Precision, Recall, F1, ROC-AUC, CM
│   ├── evaluate.py                 # Standalone test evaluation runner
│   ├── plots.py                    # ROC, Confusion Matrix, and Loss curves
│   └── comparison.py               # Baseline comparison & 5-stage ablation runner
│
├── datasets/
│   ├── dataset_loader.py           # Multi-dataset loader (ODIR, G1020, PAPILA, AIROGS, etc.)
│   └── sample_generator.py         # Realistic synthetic fundus generator for testing
│
├── utils/
│   ├── report_generator.py         # ReportLab clinical PDF report generator
│   ├── logging.py                  # Structured console & file logging
│   └── seed.py                     # Deterministic random seed controller
│
├── templates/
│   ├── base.html                   # Medical-themed base navigation layout
│   ├── index.html                  # Upload, sample test launcher, dataset status
│   ├── result.html                 # Unified multi-modal clinical report viewer
│   └── ablation.html               # Empirical comparison & ablation benchmark view
│
├── static/
│   ├── css/styles.css              # Clean medical-AI research styling
│   ├── js/app.js                   # Drag-drop upload & AJAX inference client
│   └── generated/                  # Generated heatmaps, graphs, radar charts, and PDFs
│
├── checkpoints/                    # Saved model weight checkpoints (.pth)
├── uploads/                        # Temporary uploaded images
└── results/                        # Evaluation metrics JSON and benchmark plots
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- Windows, Linux, or macOS
- (Optional) NVIDIA GPU with CUDA for accelerated training and inference. CPU fallback is fully supported.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Configuration

The system is configured in `config.py` to support major clinical retinal fundus datasets:
- **ODIR** (Ocular Disease Intelligent Recognition)
- **G1020** (Glaucoma 1020 Fundus Images)
- **PAPILA** (Papila Glaucoma Challenge)
- **AIROGS** (Artificial Intelligence for Robust Glaucoma Screening)
- **RIM-ONE DL** (Retinal Image Database for Optic Nerve Evaluation)

### Connecting Your Datasets:
Place your dataset folders in `data/<DATASET_NAME>/` or configure their environment variables:
```bash
set ODIR_DIR=C:\path\to\ODIR
set G1020_DIR=C:\path\to\G1020
set PAPILA_DIR=C:\path\to\PAPILA
```

### Out-of-the-Box Instant Testing:
If external datasets are not yet downloaded, generate a synthetic fundus testing cohort immediately:
```bash
python datasets/sample_generator.py
```
This populates `data/synthetic_fundus/` with simulated color fundus images (optic disc, physiological cup, enlarged glaucomatous CDR, vascular arcades, and macula) for instant verification.

---

## 🏋️ Training the Models

### 1. Train the Proposed X-HGNN-VLT Framework (End-to-End)
```bash
python training/train.py --model proposed --dataset synthetic --epochs 10 --batch-size 8 --lr 0.0001
```
*Saves the optimal checkpoint to `checkpoints/best_model.pth` and generates `results/training_curve_proposed.png`.*

### 2. Train the Mini-Project Baseline CNN
```bash
python training/train.py --model baseline_cnn --dataset synthetic --epochs 10 --batch-size 8 --lr 0.0001
```
*Saves the baseline checkpoint to `checkpoints/baseline_cnn.pth`.*

---

## 📈 Evaluation & Benchmark Comparison

### 1. Evaluate Model Performance
```bash
python evaluation/evaluate.py --model proposed --dataset synthetic
```
*Outputs Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and saves plots to `results/`.*

### 2. Run Baseline Comparison & 5-Stage Ablation Study
```bash
python evaluation/comparison.py
```
Runs:
- **Baseline CNN (Mini-Project)** vs. **Proposed X-HGNN-VLT**
- **5-Stage Ablation Study**:
  - `Config A`: Baseline CNN
  - `Config B`: EfficientNet-B3 Backbone
  - `Config C`: EfficientNet-B3 + X-HGNN
  - `Config D`: EfficientNet-B3 + X-HGNN + Grad-CAM
  - `Config E`: Full X-HGNN-VLT (Complete Proposed Architecture)

---

## 🌐 Launching the Web Application

Start the Flask application server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Features of the Web Interface:
1. **Interactive Upload Screen**: Drag-and-drop or select JPG, JPEG, or PNG images.
2. **Instant Test Buttons**: Click **"Load Normal Sample"** or **"Load Glaucoma Sample"** to test without manual files.
3. **Unified Clinical Report**:
   - Primary Screening Assessment (**Normal** vs. **Glaucoma**) + Confidence %.
   - **Grad-CAM Visual Heatmap & Overlay**: Identifies local image regions driving the prediction.
   - **X-HGNN Decision-Relevant Sub-Hypergraph Grounding**: Spatial visualization of active feature nodes and hyperedges.
   - **Clinical Concept Radar & Table**: 5 model-derived retinal concept scores (CDR-P, rim thinning, cup excavation, etc.).
   - **Vision-Language Transformer (VLT) Narrative**: Natural-language clinical explanation.
4. **PDF Report Export**: Click **"Download Clinical Report (PDF)"** to export an official, multi-modal clinical summary.
5. **Ablation & Baseline Dashboard**: View empirical comparisons, ROC curves, and training trajectories.

---

## 🔌 REST API Documentation

### 1. `POST /predict`
Submits a fundus image for end-to-end analysis.
- **Request**: `multipart/form-data` with fields:
  - `image`: Retinal fundus image file (.jpg, .jpeg, .png)
  - `patient_id` (optional): Patient reference string
- **Response**: `application/json`
  ```json
  {
    "status": "success",
    "case_id": "A4B92F10",
    "prediction": "Glaucoma",
    "confidence": 0.884,
    "gradcam_url": "/static/generated/A4B92F10_gradcam.png",
    "hgnn_url": "/static/generated/A4B92F10_hgnn.png",
    "radar_url": "/static/generated/A4B92F10_radar.png",
    "pdf_url": "/download_pdf/A4B92F10",
    "report_url": "/report/A4B92F10",
    "clinical_concepts": [
      {
        "title": "Cup-to-Disc Ratio Proxy (CDR-P)",
        "score": 0.742,
        "status": "Suspect / Elevated",
        "description": "Estimated CDR proxy > 0.55 (enlarged vertical/horizontal cup)."
      }
    ],
    "vlt_explanation": "Screening Result: Glaucoma Suspect...",
    "safety_notice": "AI-assisted screening result..."
  }
  ```

### 2. `GET /health`
Returns system status, device, checkpoint state, and supported classes.

### 3. `GET /download_pdf/<case_id>`
Downloads the ReportLab-generated PDF report for the case.

---

## ⚖️ Research Integrity & Medical Safety

- **Non-Fabrication**: All metrics (Accuracy, Precision, Recall, F1, ROC-AUC), Grad-CAM heatmaps, hypergraph selection weights, and clinical concept scores are computed directly by running actual forward and backward passes on the PyTorch models.
- **Labeling of Concepts**: All clinical concept scores are explicitly designated as **"model-derived concepts"**, ensuring users understand they are neural network feature proxies rather than physical micrometer anatomical measurements.
- **Medical Disclaimer**:
  > *"AI-assisted screening result. This system is a research prototype and is not a substitute for professional ophthalmological examination or clinical diagnosis."*
