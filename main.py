import os
import sys
import uuid
import datetime
import json
import cv2
import numpy as np
import torch
from typing import Optional

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from config import Config
from utils.seed import set_seed
from utils.logging import logger
from database import (
    init_db, save_prediction_record, get_prediction_record,
    list_prediction_records, get_database_stats
)
from datasets.dataset_loader import check_all_datasets
from datasets.sample_generator import generate_synthetic_fundus_image
from preprocessing.preprocessing import preprocess_for_inference, ALLOWED_EXTENSIONS
from models.full_model import ExplainableGlaucomaModel
from training.checkpoints import load_checkpoint, checkpoint_exists
from explainability.gradcam import GradCAM
from explainability.hypergraph_explainer import HypergraphExplainer
from explainability.clinical_concepts import ClinicalConceptManager
from explainability.vlt_explainer import VLTExplainer
from utils.report_generator import MedicalReportPDFGenerator
from evaluation.comparison import run_baseline_comparison, run_ablation_study
from services.gemini_service import GeminiConsultationService

# Ensure storage directories exist
for d in [Config.UPLOAD_DIR, Config.RESULTS_DIR, Config.GENERATED_DIR, Config.CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# Initialize database
init_db()

# Set reproducible seed
set_seed(Config.SEED)

# Initialize FastAPI App
app = FastAPI(
    title="GLUCOM — Explainable Glaucoma Detection Platform",
    description="Multimodal clinical screening powered by EfficientNet-B3, X-HGNN, Gumbel-Softmax, and VLT.",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 Templates with Flask-compatible url_for support
templates = Jinja2Templates(directory="templates")

# Initialize Model & Services
logger.info(f"Initializing Explainable Glaucoma Detection Model on {Config.DEVICE}...")
model = ExplainableGlaucomaModel(
    num_classes=Config.NUM_CLASSES,
    pretrained=False,
    fine_tune=False
)

checkpoint_loaded = False
if checkpoint_exists(Config.BEST_MODEL_PATH):
    try:
        load_checkpoint(Config.BEST_MODEL_PATH, model, device=Config.DEVICE)
        checkpoint_loaded = True
        logger.info(f"Successfully loaded model weights from {Config.BEST_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}. Model will use initialized weights.")
else:
    logger.warning("No checkpoint found at startup. Running with initialized weights.")

model = model.to(Config.DEVICE)
model.eval()

# Explainability & AI Services
gradcam_service = GradCAM(model)
vlt_service = VLTExplainer(model.vlt)
gemini_service = GeminiConsultationService()

# Security helper for magic bytes validation
def validate_image_header(file_bytes: bytes) -> bool:
    if len(file_bytes) < 4:
        return False
    # JPEG magic bytes: FF D8 FF
    if file_bytes[:3] == b"\xff\xd8\xff":
        return True
    # PNG magic bytes: 89 50 4E 47
    if file_bytes[:4] == b"\x89PNG":
        return True
    return False

# -------------------------------------------------------------
# Web & Dashboard Routes (100% Preserved UI)
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_view(request: Request):
    dataset_report = check_all_datasets()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "device": Config.DEVICE,
            "checkpoint_loaded": checkpoint_loaded,
            "dataset_report": dataset_report
        }
    )

@app.get("/report/{case_id}", response_class=HTMLResponse)
def view_report(request: Request, case_id: str):
    data = get_prediction_record(case_id)
    if not data:
        # Check if PDF exists on disk as fallback
        pdf_path = os.path.join(Config.GENERATED_DIR, f"{case_id}_clinical_report.pdf")
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Screening report not found or expired.")
        data = {
            "case_id": case_id,
            "patient_ref": "ANON-SCREENING",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "prediction": "Unknown",
            "confidence": 0.5,
            "original_img": f"{case_id}_prep.png",
            "gradcam_img": f"{case_id}_gradcam.png",
            "hgnn_img": f"{case_id}_hgnn.png",
            "radar_img": f"{case_id}_radar.png",
            "concepts": [],
            "vlt_narrative": "Historical case report loaded from disk archive.",
            "vlt_findings": [],
            "safety_notice": Config.DISCLAIMER_TEXT
        }

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "case_id": data["case_id"],
            "patient_ref": data.get("patient_ref", "ANON-SCREENING"),
            "timestamp": data.get("timestamp"),
            "prediction": data.get("prediction"),
            "confidence": data.get("confidence"),
            "original_img": data.get("original_img"),
            "gradcam_img": data.get("gradcam_img"),
            "hgnn_img": data.get("hgnn_img"),
            "radar_img": data.get("radar_img"),
            "concepts": data.get("concepts", []),
            "vlt_narrative": data.get("vlt_narrative", ""),
            "vlt_findings": data.get("vlt_findings", []),
            "safety_notice": data.get("safety_notice", Config.DISCLAIMER_TEXT)
        }
    )

@app.get("/ablation", response_class=HTMLResponse)
def ablation_view(request: Request):
    comp_file = os.path.join(Config.RESULTS_DIR, "baseline_comparison.json")
    ablation_file = os.path.join(Config.RESULTS_DIR, "ablation_study.json")

    comparison = None
    if os.path.exists(comp_file):
        try:
            with open(comp_file, "r", encoding="utf-8") as f:
                comparison = json.load(f)
        except Exception:
            comparison = None

    ablations = []
    if os.path.exists(ablation_file):
        try:
            with open(ablation_file, "r", encoding="utf-8") as f:
                ablations = json.load(f)
        except Exception:
            ablations = []

    roc_path = os.path.join(Config.RESULTS_DIR, "roc_curve_proposed.png")
    curve_path = os.path.join(Config.RESULTS_DIR, "training_curve_proposed.png")

    return templates.TemplateResponse(
        "ablation.html",
        {
            "request": request,
            "comparison": comparison,
            "ablations": ablations,
            "roc_exists": os.path.exists(roc_path),
            "curve_exists": os.path.exists(curve_path)
        }
    )

# -------------------------------------------------------------
# Core Production REST API Endpoints
# -------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": Config.DEVICE,
        "cuda_available": (Config.DEVICE == "cuda"),
        "model_architecture": "EfficientNet-B3 + X-HGNN + Gumbel-Softmax + VLT",
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": Config.BEST_MODEL_PATH,
        "num_classes": Config.NUM_CLASSES,
        "classes": Config.CLASS_NAMES,
        "database_status": "connected",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/model/info")
def model_info():
    metrics_path = os.path.join(Config.RESULTS_DIR, "metrics_proposed.json")
    latest_metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                latest_metrics = json.load(f)
        except Exception:
            pass

    return {
        "name": "GLUCOM Explainable Glaucoma Detection Platform",
        "version": "2.0.0",
        "backbone": {
            "name": Config.BACKBONE_NAME,
            "out_channels": Config.BACKBONE_OUT_CHANNELS,
            "pretrained": Config.PRETRAINED
        },
        "hypergraph": {
            "num_nodes": Config.NUM_NODES,
            "grid_size": list(Config.GRID_SIZE),
            "node_embedding_dim": Config.NODE_EMB_DIM,
            "total_hyperedges": Config.TOTAL_HYPEREDGES,
            "spatial_hyperedges": Config.NUM_SPATIAL_HYPEREDGES,
            "semantic_hyperedges": Config.NUM_SEMANTIC_HYPEREDGES,
            "vascular_hyperedges": Config.NUM_VASCULAR_HYPEREDGES,
            "hgnn_hidden_dim": Config.HGNN_HIDDEN_DIM,
            "hgnn_out_dim": Config.HGNN_OUT_DIM
        },
        "selector": {
            "method": "Gumbel-Softmax",
            "top_k_selected": Config.SUBGRAPH_TOP_K
        },
        "clinical_concepts": Config.CLINICAL_CONCEPTS,
        "vision_language_transformer": {
            "hidden_dim": Config.VLT_HIDDEN_DIM,
            "num_heads": Config.VLT_NUM_HEADS,
            "num_layers": Config.VLT_NUM_LAYERS
        },
        "evaluation_metrics": latest_metrics,
        "classes": Config.CLASS_NAMES,
        "disclaimer": Config.DISCLAIMER_TEXT
    }

@app.get("/api/sample/{sample_type}")
def get_sample_image(sample_type: str):
    is_glaucoma = (sample_type.lower() == "glaucoma")
    img = generate_synthetic_fundus_image(is_glaucoma=is_glaucoma)
    temp_path = os.path.join(Config.UPLOAD_DIR, f"temp_sample_{sample_type}.png")
    cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return FileResponse(temp_path, media_type="image/png")

@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    patient_id: Optional[str] = Form("ANON-SCREENING")
):
    if not image or not image.filename:
        raise HTTPException(status_code=400, detail="No image file provided in upload request.")

    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    contents = image.file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > 32 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded image exceeds 32MB limit.")
    if not validate_image_header(contents):
        raise HTTPException(status_code=400, detail="File content does not match a valid JPEG or PNG image header.")

    case_id = str(uuid.uuid4())[:8].upper()
    patient_ref = (patient_id or "ANON-SCREENING").strip()

    # Save uploaded raw image
    upload_path = os.path.join(Config.UPLOAD_DIR, f"{case_id}_original{ext}")
    with open(upload_path, "wb") as f:
        f.write(contents)

    try:
        # Step 1: Preprocessing & CLAHE enhancement
        tensor, enhanced_rgb = preprocess_for_inference(upload_path, image_size=Config.IMAGE_SIZE)
        tensor = tensor.to(Config.DEVICE)

        prep_filename = f"{case_id}_prep.png"
        prep_path = os.path.join(Config.GENERATED_DIR, prep_filename)
        cv2.imwrite(prep_path, cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))

        # Step 2: Model Forward Pass
        with torch.no_grad():
            outputs = model(tensor, temperature=1.0, hard_selection=False)

        pred_idx = outputs["predicted_class"].item()
        confidence = float(outputs["confidence"].item())
        prediction_label = Config.CLASS_NAMES[pred_idx]

        # Step 3: Grad-CAM Heatmap & Overlay
        cam_heatmap, _, _ = gradcam_service.generate_heatmap(tensor, target_class=pred_idx)
        cam_overlay, _ = gradcam_service.create_overlay(enhanced_rgb, cam_heatmap)
        
        gradcam_filename = f"{case_id}_gradcam.png"
        gradcam_path = os.path.join(Config.GENERATED_DIR, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

        # Step 4: X-HGNN Sub-Hypergraph Grounding
        coords = outputs["coords"]
        H_sub = outputs["H_sub"][0]
        edge_weights = outputs["edge_weights"][0]
        selected_topk = outputs["selected_topk"][0]

        hgnn_filename = f"{case_id}_hgnn.png"
        hgnn_path = os.path.join(Config.GENERATED_DIR, hgnn_filename)
        HypergraphExplainer.visualize_sub_hypergraph(
            image_rgb=enhanced_rgb,
            coords=coords,
            H_sub=H_sub,
            edge_weights=edge_weights,
            selected_topk=selected_topk,
            save_path=hgnn_path
        )

        # Step 5: Model-Derived Clinical Concepts & Radar Chart
        raw_concept_scores = outputs["concept_scores"]
        formatted_concepts = ClinicalConceptManager.format_concepts_for_ui(raw_concept_scores)
        
        radar_filename = f"{case_id}_radar.png"
        radar_path = os.path.join(Config.GENERATED_DIR, radar_filename)
        ClinicalConceptManager.generate_radar_chart(raw_concept_scores, save_path=radar_path)

        # Step 6: Vision-Language Transformer (VLT) Narrative
        vlt_explanation_dict = vlt_service.explain(
            prediction_label=prediction_label,
            confidence=confidence,
            concept_scores=raw_concept_scores
        )
        vlt_narrative = vlt_explanation_dict["narrative"]

        # Step 7: Clinical PDF Report
        pdf_filename = f"{case_id}_clinical_report.pdf"
        pdf_path = os.path.join(Config.GENERATED_DIR, pdf_filename)
        MedicalReportPDFGenerator.generate_pdf_report(
            output_pdf_path=pdf_path,
            case_id=case_id,
            prediction_label=prediction_label,
            confidence=confidence,
            original_img_path=prep_path,
            gradcam_img_path=gradcam_path,
            hgnn_img_path=hgnn_path,
            radar_img_path=radar_path,
            clinical_concepts_list=formatted_concepts,
            vlt_explanation_text=vlt_narrative,
            patient_ref=patient_ref
        )

        # Step 8: Persistent Database Storage
        report_data = {
            "case_id": case_id,
            "patient_ref": patient_ref,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "prediction": prediction_label,
            "confidence": confidence,
            "original_img": prep_filename,
            "gradcam_img": gradcam_filename,
            "hgnn_img": hgnn_filename,
            "radar_img": radar_filename,
            "pdf_filename": pdf_filename,
            "concepts": formatted_concepts,
            "vlt_narrative": vlt_narrative,
            "vlt_findings": vlt_explanation_dict.get("key_findings", []),
            "safety_notice": Config.DISCLAIMER_TEXT
        }
        save_prediction_record(report_data)

        # Response matching exact frontend contract
        return {
            "status": "success",
            "case_id": case_id,
            "patient_ref": patient_ref,
            "prediction": prediction_label,
            "confidence": round(confidence, 4),
            "report_url": f"/report/{case_id}",
            "pdf_url": f"/download_pdf/{case_id}",
            "orig_url": f"/static/generated/{prep_filename}",
            "gradcam_url": f"/static/generated/{gradcam_filename}",
            "hgnn_url": f"/static/generated/{hgnn_filename}",
            "radar_url": f"/static/generated/{radar_filename}",
            "clinical_concepts": formatted_concepts,
            "vlt_explanation": vlt_narrative,
            "safety_notice": Config.DISCLAIMER_TEXT
        }

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Error during model inference: {str(e)}")

@app.post("/explain")
def explain_case(
    case_id: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    if case_id:
        record = get_prediction_record(case_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Case ID '{case_id}' not found.")
        return {
            "case_id": case_id,
            "prediction": record["prediction"],
            "confidence": record["confidence"],
            "gradcam_url": f"/static/generated/{record['gradcam_img']}",
            "hgnn_url": f"/static/generated/{record['hgnn_img']}",
            "radar_url": f"/static/generated/{record['radar_img']}",
            "clinical_concepts": record["concepts"],
            "vlt_narrative": record["vlt_narrative"]
        }
    elif image:
        # Re-run predict pipeline for image
        return predict(image=image, patient_id="EXPLAIN-QUERY")
    else:
        raise HTTPException(status_code=400, detail="Must provide either case_id or image for explanation.")

@app.get("/history")
def history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    records = list_prediction_records(limit=limit, offset=offset)
    stats = get_database_stats()
    return {
        "total_records": stats["total_screenings"],
        "returned": len(records),
        "limit": limit,
        "offset": offset,
        "stats": stats,
        "history": records
    }

@app.get("/history/{case_id}")
def get_history_item(case_id: str):
    record = get_prediction_record(case_id)
    if not record:
        raise HTTPException(status_code=404, detail="Case record not found.")
    return record

@app.get("/download_pdf/{case_id}")
def download_pdf(case_id: str):
    pdf_filename = f"{case_id}_clinical_report.pdf"
    pdf_path = os.path.join(Config.GENERATED_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report PDF file not found.")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"Glaucoma_Screening_Report_{case_id}.pdf"
    )

@app.post("/api/run_comparison")
def trigger_comparison():
    try:
        run_baseline_comparison()
        run_ablation_study()
    except Exception as e:
        logger.error(f"Error running comparison: {e}")
    return RedirectResponse(url="/ablation", status_code=303)

@app.get("/api/stats")
def api_stats():
    return get_database_stats()

@app.post("/api/gemini/consult")
def gemini_consult(
    case_id: str = Form(...),
    query: Optional[str] = Form(None)
):
    record = get_prediction_record(case_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Case ID '{case_id}' not found.")
    
    result = gemini_service.consult(
        case_id=case_id,
        prediction=record["prediction"],
        confidence=record["confidence"],
        concepts=record.get("concepts", []),
        vlt_narrative=record.get("vlt_narrative", ""),
        query=query
    )
    return result

# -------------------------------------------------------------
# Standalone Runner
# -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Glucom FastAPI Production Backend on http://127.0.0.1:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

