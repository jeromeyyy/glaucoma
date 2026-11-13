import os
import sys
import uuid
import datetime
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for

from config import Config
from utils.seed import set_seed
from utils.logging import logger
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

# Initialize application
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB limit
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Ensure required directories exist
for d in [Config.UPLOAD_DIR, Config.RESULTS_DIR, Config.GENERATED_DIR, Config.CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# Global in-memory cache for recent case reports
REPORTS_CACHE = {}

# Set reproducible seed
set_seed(Config.SEED)

# -------------------------------------------------------------
# Model Initialization (Loaded once at startup)
# -------------------------------------------------------------
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
        logger.info(f"Successfully loaded trained weights from {Config.BEST_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}. Model will use initialized weights.")
else:
    logger.warning("No checkpoint found at startup. Running with initialized weights.")

model = model.to(Config.DEVICE)
model.eval()

# Explainability Services
gradcam_service = GradCAM(model)
vlt_service = VLTExplainer(model.vlt)

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    dataset_report = check_all_datasets()
    return render_template(
        "index.html",
        device=Config.DEVICE,
        checkpoint_loaded=checkpoint_loaded,
        dataset_report=dataset_report
    )

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "device": Config.DEVICE,
        "cuda_available": (Config.DEVICE == "cuda"),
        "model_architecture": "EfficientNet-B3 + X-HGNN + Gumbel-Softmax + VLT",
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_path": Config.BEST_MODEL_PATH,
        "num_classes": Config.NUM_CLASSES,
        "classes": Config.CLASS_NAMES,
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route("/api/sample/<sample_type>")
def get_sample_image(sample_type):
    """
    Returns a sample retinal fundus image for one-click testing.
    """
    is_glaucoma = (sample_type.lower() == "glaucoma")
    img = generate_synthetic_fundus_image(is_glaucoma=is_glaucoma)
    temp_path = os.path.join(Config.UPLOAD_DIR, f"temp_sample_{sample_type}.png")
    cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return send_file(temp_path, mimetype="image/png")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in upload request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Invalid format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    case_id = str(uuid.uuid4())[:8].upper()
    patient_ref = request.form.get("patient_id", "ANON-SCREENING").strip() or "ANON-SCREENING"

    # Save uploaded image
    upload_path = os.path.join(Config.UPLOAD_DIR, f"{case_id}_original{ext}")
    file.save(upload_path)

    try:
        # Step 1: Preprocessing & CLAHE enhancement
        tensor, enhanced_rgb = preprocess_for_inference(upload_path, image_size=Config.IMAGE_SIZE)
        tensor = tensor.to(Config.DEVICE)

        # Save preprocessed fundus image for visualizer
        prep_filename = f"{case_id}_prep.png"
        prep_path = os.path.join(Config.GENERATED_DIR, prep_filename)
        cv2.imwrite(prep_path, cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))

        # Step 2: Model Forward Pass
        import torch
        with torch.no_grad():
            outputs = model(tensor, temperature=1.0, hard_selection=False)

        pred_idx = outputs["predicted_class"].item()
        confidence = outputs["confidence"].item()
        prediction_label = Config.CLASS_NAMES[pred_idx]

        # Step 3: Grad-CAM Heatmap & Overlay
        cam_heatmap, _, _ = gradcam_service.generate_heatmap(tensor, target_class=pred_idx)
        cam_overlay, _ = gradcam_service.create_overlay(enhanced_rgb, cam_heatmap)
        
        gradcam_filename = f"{case_id}_gradcam.png"
        gradcam_path = os.path.join(Config.GENERATED_DIR, gradcam_filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

        # Step 4: X-HGNN Sub-Hypergraph Grounding Visualization
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

        # Step 6: Vision-Language Transformer (VLT) Clinical Explanation
        vlt_explanation_dict = vlt_service.explain(
            prediction_label=prediction_label,
            confidence=confidence,
            concept_scores=raw_concept_scores
        )
        vlt_narrative = vlt_explanation_dict["narrative"]

        # Step 7: Multi-Modal PDF Report Generation
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

        # Cache results for report viewer
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
            "vlt_findings": vlt_explanation_dict["key_findings"],
            "safety_notice": Config.DISCLAIMER_TEXT
        }
        REPORTS_CACHE[case_id] = report_data

        # If AJAX JSON requested:
        return jsonify({
            "status": "success",
            "case_id": case_id,
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
        })

    except Exception as e:
        logger.exception("Inference failed")
        return jsonify({"error": f"Error during model inference: {str(e)}"}), 500

@app.route("/report/<case_id>")
def view_report(case_id):
    if case_id not in REPORTS_CACHE:
        # Check if files exist on disk
        pdf_path = os.path.join(Config.GENERATED_DIR, f"{case_id}_clinical_report.pdf")
        if not os.path.exists(pdf_path):
            return render_template("base.html", content="Report not found or expired."), 404

    data = REPORTS_CACHE[case_id]
    return render_template(
        "result.html",
        case_id=data["case_id"],
        patient_ref=data["patient_ref"],
        timestamp=data["timestamp"],
        prediction=data["prediction"],
        confidence=data["confidence"],
        original_img=data["original_img"],
        gradcam_img=data["gradcam_img"],
        hgnn_img=data["hgnn_img"],
        radar_img=data["radar_img"],
        concepts=data["concepts"],
        vlt_narrative=data["vlt_narrative"],
        vlt_findings=data["vlt_findings"],
        safety_notice=data["safety_notice"]
    )

@app.route("/download_pdf/<case_id>")
def download_pdf(case_id):
    pdf_filename = f"{case_id}_clinical_report.pdf"
    pdf_path = os.path.join(Config.GENERATED_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": "Report PDF not found."}), 404
    return send_file(pdf_path, as_attachment=True, download_name=f"Glaucoma_Screening_Report_{case_id}.pdf")

@app.route("/ablation")
def ablation_view():
    comp_file = os.path.join(Config.RESULTS_DIR, "baseline_comparison.json")
    ablation_file = os.path.join(Config.RESULTS_DIR, "ablation_study.json")

    comparison = None
    if os.path.exists(comp_file):
        with open(comp_file, "r", encoding="utf-8") as f:
            comparison = json.load(f)

    ablations = []
    if os.path.exists(ablation_file):
        with open(ablation_file, "r", encoding="utf-8") as f:
            ablations = json.load(f)

    roc_path = os.path.join(Config.RESULTS_DIR, "roc_curve_proposed.png")
    curve_path = os.path.join(Config.RESULTS_DIR, "training_curve_proposed.png")

    return render_template(
        "ablation.html",
        comparison=comparison,
        ablations=ablations,
        roc_exists=os.path.exists(roc_path),
        curve_exists=os.path.exists(curve_path)
    )

@app.route("/api/run_comparison", methods=["POST"])
def trigger_comparison():
    try:
        run_baseline_comparison()
        run_ablation_study()
    except Exception as e:
        logger.error(f"Error running comparison: {e}")
    return redirect(url_for("ablation_view"))

@app.route("/static/results/<filename>")
def serve_results_file(filename):
    return send_file(os.path.join(Config.RESULTS_DIR, filename))

# -------------------------------------------------------------
# Main Application Launcher
# -------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Glaucoma Detection Flask App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
