import os
import sqlite3
import json
import datetime
from config import Config

DB_PATH = os.path.join(Config.BASE_DIR, "data", "glucom.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        case_id TEXT PRIMARY KEY,
        patient_ref TEXT,
        timestamp TEXT,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        original_img TEXT,
        gradcam_img TEXT,
        hgnn_img TEXT,
        radar_img TEXT,
        pdf_filename TEXT,
        concepts_json TEXT,
        vlt_narrative TEXT,
        vlt_findings_json TEXT,
        doctor_notes TEXT,
        feedback_status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_ref);
    """)
    conn.commit()
    conn.close()

def save_prediction_record(data: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO predictions (
        case_id, patient_ref, timestamp, prediction, confidence,
        original_img, gradcam_img, hgnn_img, radar_img, pdf_filename,
        concepts_json, vlt_narrative, vlt_findings_json, doctor_notes, feedback_status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("case_id"),
        data.get("patient_ref", "ANON-SCREENING"),
        data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")),
        data.get("prediction"),
        float(data.get("confidence", 0.0)),
        data.get("original_img"),
        data.get("gradcam_img"),
        data.get("hgnn_img"),
        data.get("radar_img"),
        data.get("pdf_filename"),
        json.dumps(data.get("concepts", {})),
        data.get("vlt_narrative", ""),
        json.dumps(data.get("vlt_findings", [])),
        data.get("doctor_notes", ""),
        data.get("feedback_status", "")
    ))
    conn.commit()
    conn.close()

def get_prediction_record(case_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE case_id = ?", (case_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d["concepts"] = json.loads(d.get("concepts_json") or "{}")
    except Exception:
        d["concepts"] = {}
    try:
        d["vlt_findings"] = json.loads(d.get("vlt_findings_json") or "[]")
    except Exception:
        d["vlt_findings"] = []
    d["safety_notice"] = Config.DISCLAIMER_TEXT
    return d

def list_prediction_records(limit=50, offset=0):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT case_id, patient_ref, timestamp, prediction, confidence, pdf_filename, original_img, gradcam_img, created_at
    FROM predictions
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?
    """, (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_database_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Glaucoma'")
    glaucoma_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Normal'")
    normal_count = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = cursor.fetchone()[0] or 0.0
    conn.close()
    return {
        "total_screenings": total,
        "glaucoma_suspects": glaucoma_count,
        "normal_cases": normal_count,
        "average_confidence": round(float(avg_conf), 4)
    }

# Initialize on import
init_db()

