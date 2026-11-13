import os
import sys
import tempfile
import numpy as np
import cv2
import torch
import io
from fastapi import UploadFile, HTTPException

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from database import (
    init_db, save_prediction_record, get_prediction_record,
    list_prediction_records, get_database_stats
)
from datasets.dataset_loader import scan_dataset_directory, _stratified_split
from models.full_model import ExplainableGlaucomaModel
from explainability.gradcam import GradCAM
from explainability.hypergraph_explainer import HypergraphExplainer
from explainability.clinical_concepts import ClinicalConceptManager
from explainability.vlt_explainer import VLTExplainer
from utils.report_generator import MedicalReportPDFGenerator
import main
from main import validate_image_header

# -------------------------------------------------------------
# 1. Database Persistence Tests
# -------------------------------------------------------------
def test_database_crud():
    init_db()
    test_case_id = "TEST-CASE-99"
    test_data = {
        "case_id": test_case_id,
        "patient_ref": "PATIENT-TEST-001",
        "timestamp": "2026-11-13 12:00:00 UTC",
        "prediction": "Glaucoma",
        "confidence": 0.9421,
        "original_img": "test_prep.png",
        "gradcam_img": "test_gradcam.png",
        "hgnn_img": "test_hgnn.png",
        "radar_img": "test_radar.png",
        "pdf_filename": "test_report.pdf",
        "concepts": [{"title": "Optic Cup", "score": 0.85, "status": "Significant Excavation"}],
        "vlt_narrative": "Test narrative generated.",
        "vlt_findings": ["Finding 1", "Finding 2"]
    }
    
    save_prediction_record(test_data)
    retrieved = get_prediction_record(test_case_id)
    assert retrieved is not None
    assert retrieved["case_id"] == test_case_id
    assert retrieved["prediction"] == "Glaucoma"
    assert abs(retrieved["confidence"] - 0.9421) < 1e-4
    assert len(retrieved["concepts"]) == 1
    assert len(retrieved["vlt_findings"]) == 2

    # Verify history list & stats
    history = list_prediction_records(limit=10)
    assert len(history) >= 1
    assert any(h["case_id"] == test_case_id for h in history)
    
    stats = get_database_stats()
    assert stats["total_screenings"] >= 1
    assert stats["glaucoma_suspects"] >= 1

# -------------------------------------------------------------
# 2. Dataset & Zero-Leakage Tests
# -------------------------------------------------------------
def test_dataset_zero_leakage():
    fake_samples = [(f"image_{i}.jpg", i % 2) for i in range(100)]
    train_s, val_s, test_s = _stratified_split(fake_samples, train_ratio=0.70, val_ratio=0.15, seed=42)
    
    train_set = {s[0] for s in train_s}
    val_set = {s[0] for s in val_s}
    test_set = {s[0] for s in test_s}
    
    # Assert mutual exclusivity
    assert len(train_set & val_set) == 0, "Leakage detected between train and val!"
    assert len(train_set & test_set) == 0, "Leakage detected between train and test!"
    assert len(val_set & test_set) == 0, "Leakage detected between val and test!"
    assert len(train_s) + len(val_s) + len(test_s) == 100

# -------------------------------------------------------------
# 3. Model Architecture & Forward Pass Tests
# -------------------------------------------------------------
def test_model_architecture_and_forward():
    model = ExplainableGlaucomaModel(pretrained=False, fine_tune=False)
    model.eval()
    x = torch.randn(2, 3, 300, 300)
    out = model(x, temperature=1.0, hard_selection=False)
    
    assert "logits" in out
    assert out["logits"].shape == (2, 2)
    assert "probabilities" in out
    assert out["probabilities"].shape == (2, 2)
    assert "coords" in out
    assert out["coords"].shape == (64, 2)
    assert "H_full" in out
    assert out["H_full"].shape == (2, 64, 40)
    assert "H_sub" in out
    assert out["H_sub"].shape == (2, 64, 40)
    assert "concept_scores" in out
    assert len(out["concept_scores"]) == 5
    assert "vlt_context" in out

# -------------------------------------------------------------
# 4. Explainable AI (Grad-CAM, HGNN, Radar, VLT) Tests
# -------------------------------------------------------------
def test_explainability_services():
    model = ExplainableGlaucomaModel(pretrained=False, fine_tune=False)
    model.eval()
    
    # Grad-CAM test
    gradcam = GradCAM(model)
    x = torch.randn(1, 3, 300, 300)
    cam_heatmap, target_cls, conf = gradcam.generate_heatmap(x, target_class=1)
    assert cam_heatmap.shape == (300, 300)
    assert 0.0 <= cam_heatmap.min() <= cam_heatmap.max() <= 1.0
    
    dummy_rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    overlay, _ = gradcam.create_overlay(dummy_rgb, cam_heatmap)
    assert overlay.shape == (300, 300, 3)

    # VLT Explanation test
    vlt_service = VLTExplainer(model.vlt)
    dummy_concept_scores = {
        name: torch.tensor([[0.75]]) for name in Config.CLINICAL_CONCEPTS
    }
    explanation = vlt_service.explain(
        prediction_label="Glaucoma",
        confidence=0.88,
        concept_scores=dummy_concept_scores
    )
    assert "narrative" in explanation
    assert len(explanation["narrative"]) > 50
    assert "key_findings" in explanation

    # Radar chart generation test
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        radar_path = tmp.name
    try:
        ClinicalConceptManager.generate_radar_chart(dummy_concept_scores, save_path=radar_path)
        assert os.path.exists(radar_path)
        assert os.path.getsize(radar_path) > 1000
    finally:
        if os.path.exists(radar_path):
            os.remove(radar_path)

# -------------------------------------------------------------
# 5. Security & Image Header Validation Tests
# -------------------------------------------------------------
def test_image_header_validation():
    valid_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    valid_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    invalid_php = b"<?php echo 'malicious'; ?>"
    empty = b""

    assert validate_image_header(valid_jpeg) is True
    assert validate_image_header(valid_png) is True
    assert validate_image_header(invalid_php) is False
    assert validate_image_header(empty) is False

# -------------------------------------------------------------
# 6. REST API Endpoints Tests
# -------------------------------------------------------------
def test_api_health_and_info():
    data = main.health()
    assert data["status"] == "healthy"
    assert "checkpoint_loaded" in data
    assert data["num_classes"] == 2

    info = main.model_info()
    assert info["backbone"]["name"] == "efficientnet_b3"
    assert info["hypergraph"]["total_hyperedges"] == 40
    assert len(info["clinical_concepts"]) == 5

def test_api_sample_generation():
    res_normal = main.get_sample_image("normal")
    assert os.path.exists(res_normal.path)

    res_glaucoma = main.get_sample_image("glaucoma")
    assert os.path.exists(res_glaucoma.path)

def test_api_predict_and_error_handling():
    # 1. Test empty upload error
    empty_file = UploadFile(filename="empty.jpg", file=io.BytesIO(b""))
    try:
        main.predict(image=empty_file)
        assert False, "Should have raised HTTPException for empty file"
    except HTTPException as e:
        assert e.status_code == 400

    # 2. Test invalid extension
    bad_file = UploadFile(filename="script.py", file=io.BytesIO(b"print(1)"))
    try:
        main.predict(image=bad_file)
        assert False, "Should have raised HTTPException for invalid extension"
    except HTTPException as e:
        assert e.status_code == 400

    # 3. Test genuine image upload
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 120, 50), -1)
    _, encoded = cv2.imencode(".png", img)
    img_bytes = encoded.tobytes()

    good_file = UploadFile(filename="test_fundus.png", file=io.BytesIO(img_bytes))
    pred_data = main.predict(image=good_file, patient_id="TEST-UNIT-RUNNER")
    
    assert pred_data["status"] == "success"
    assert "case_id" in pred_data
    assert pred_data["prediction"] in ["Normal", "Glaucoma"]
    assert 0.0 <= pred_data["confidence"] <= 1.0
    assert "gradcam_url" in pred_data
    assert "hgnn_url" in pred_data
    assert "radar_url" in pred_data
    assert "vlt_explanation" in pred_data

    # 4. Test explain endpoint with case_id
    explain_data = main.explain_case(case_id=pred_data["case_id"])
    assert explain_data["case_id"] == pred_data["case_id"]
    assert "gradcam_url" in explain_data

def test_model_unavailable_protection():
    """
    Verifies that the application strictly prohibits predictions
    if the trained checkpoint is offline or unverified.
    """
    original_state = main.checkpoint_loaded
    try:
        main.checkpoint_loaded = False
        dummy_file = UploadFile(filename="test.png", file=io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64))
        try:
            main.predict(image=dummy_file)
            assert False, "Should have raised HTTP 503 when checkpoint is not loaded"
        except HTTPException as e:
            assert e.status_code == 503
            assert "offline or unverified" in e.detail
    finally:
        main.checkpoint_loaded = original_state

def test_image_dependent_inference():
    """
    Verifies that distinct input images produce distinct, non-identical outputs,
    proving the model is genuine and image-dependent.
    """
    # Create two visually distinct synthetic fundus images
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img1, (150, 150), 40, (180, 220, 255), -1) # bright optic disc
    _, enc1 = cv2.imencode(".png", img1)
    file1 = UploadFile(filename="img1.png", file=io.BytesIO(enc1.tobytes()))

    img2 = np.full((300, 300, 3), 120, dtype=np.uint8)
    cv2.circle(img2, (100, 200), 80, (20, 50, 200), -1) # dark offset lesion
    _, enc2 = cv2.imencode(".png", img2)
    file2 = UploadFile(filename="img2.png", file=io.BytesIO(enc2.tobytes()))

    res1 = main.predict(image=file1, patient_id="PATIENT-ALPHA")
    res2 = main.predict(image=file2, patient_id="PATIENT-BETA")

    # Cases must have unique identifiers
    assert res1["case_id"] != res2["case_id"]
    # Image files must be unique
    assert res1["gradcam_url"] != res2["gradcam_url"]
    assert res1["hgnn_url"] != res2["hgnn_url"]
    # Concept representations must be image-specific (not all identical)
    c1 = [c["score"] for c in res1["clinical_concepts"]]
    c2 = [c["score"] for c in res2["clinical_concepts"]]
    assert c1 != c2, f"Clinical concept scores should depend on the input image: {c1} vs {c2}"

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    tests = [
        ("Database Persistence (CRUD)", test_database_crud),
        ("Dataset & Zero-Leakage Validation", test_dataset_zero_leakage),
        ("Model Architecture & Forward Pass", test_model_architecture_and_forward),
        ("Explainable AI Services (Grad-CAM, HGNN, Radar, VLT)", test_explainability_services),
        ("Security & Image Header Magic Bytes", test_image_header_validation),
        ("API Health & Model Info Endpoints", test_api_health_and_info),
        ("API Sample Generation Endpoints", test_api_sample_generation),
        ("API Prediction, Explanation & Error Handling", test_api_predict_and_error_handling),
        ("Model-Unavailable Protection & Safety", test_model_unavailable_protection),
        ("Image-Dependent Inference & Non-Identical Outputs", test_image_dependent_inference),
    ]
    
    print("\n" + "=" * 60, flush=True)
    print("      RUNNING GLUCOM AUTOMATED TEST SUITE", flush=True)
    print("=" * 60, flush=True)
    
    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"[*] Running: {name} ...", end=" ", flush=True)
            test_fn()
            print("PASSED [✓]", flush=True)
            passed += 1
        except Exception as e:
            print(f"FAILED [✗] ({e})", flush=True)
            import traceback
            traceback.print_exc()
            failed += 1
            
    print("=" * 60, flush=True)
    print(f"TOTAL: {len(tests)} | PASSED: {passed} | FAILED: {failed}", flush=True)
    print("=" * 60, flush=True)
    if failed > 0:
        sys.exit(1)


