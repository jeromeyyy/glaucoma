import os
import sqlite3
import json
import datetime
from config import Config
from utils.logging import logger

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_PATH = os.path.join(Config.BASE_DIR, "data", "glucom.db")

def is_postgres():
    return bool(DATABASE_URL and (DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")))

def get_db_connection():
    if is_postgres():
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            url = DATABASE_URL
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
            return conn, "postgres"
        except Exception as e:
            logger.warning(f"Failed to connect to PostgreSQL via DATABASE_URL ({e}). Falling back to SQLite.")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn, "sqlite"

def init_db():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    if db_type == "postgres":
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            case_id VARCHAR(128) PRIMARY KEY,
            patient_ref VARCHAR(128),
            timestamp VARCHAR(128),
            prediction VARCHAR(64) NOT NULL,
            confidence REAL NOT NULL,
            original_img VARCHAR(256),
            gradcam_img VARCHAR(256),
            hgnn_img VARCHAR(256),
            radar_img VARCHAR(256),
            pdf_filename VARCHAR(256),
            concepts_json TEXT,
            vlt_narrative TEXT,
            vlt_findings_json TEXT,
            doctor_notes TEXT,
            feedback_status VARCHAR(64),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_ref);")
    else:
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_ref);")
    conn.commit()
    conn.close()

def save_prediction_record(data: dict):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    values = (
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
    )

    if db_type == "postgres":
        cursor.execute("""
        INSERT INTO predictions (
            case_id, patient_ref, timestamp, prediction, confidence,
            original_img, gradcam_img, hgnn_img, radar_img, pdf_filename,
            concepts_json, vlt_narrative, vlt_findings_json, doctor_notes, feedback_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (case_id) DO UPDATE SET
            patient_ref = EXCLUDED.patient_ref,
            timestamp = EXCLUDED.timestamp,
            prediction = EXCLUDED.prediction,
            confidence = EXCLUDED.confidence,
            original_img = EXCLUDED.original_img,
            gradcam_img = EXCLUDED.gradcam_img,
            hgnn_img = EXCLUDED.hgnn_img,
            radar_img = EXCLUDED.radar_img,
            pdf_filename = EXCLUDED.pdf_filename,
            concepts_json = EXCLUDED.concepts_json,
            vlt_narrative = EXCLUDED.vlt_narrative,
            vlt_findings_json = EXCLUDED.vlt_findings_json,
            doctor_notes = EXCLUDED.doctor_notes,
            feedback_status = EXCLUDED.feedback_status;
        """, values)
    else:
        cursor.execute("""
        INSERT OR REPLACE INTO predictions (
            case_id, patient_ref, timestamp, prediction, confidence,
            original_img, gradcam_img, hgnn_img, radar_img, pdf_filename,
            concepts_json, vlt_narrative, vlt_findings_json, doctor_notes, feedback_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values)

    conn.commit()
    conn.close()

def get_prediction_record(case_id: str):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM predictions WHERE case_id = %s" if db_type == "postgres" else "SELECT * FROM predictions WHERE case_id = ?"
    cursor.execute(query, (case_id,))
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
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT case_id, patient_ref, timestamp, prediction, confidence, pdf_filename, original_img, gradcam_img, created_at
    FROM predictions
    ORDER BY created_at DESC
    LIMIT %s OFFSET %s
    """ if db_type == "postgres" else """
    SELECT case_id, patient_ref, timestamp, prediction, confidence, pdf_filename, original_img, gradcam_img, created_at
    FROM predictions
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?
    """
    cursor.execute(query, (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_database_stats():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0] if db_type == "sqlite" else cursor.fetchone()["count"]
    
    if db_type == "sqlite":
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Glaucoma'")
        glaucoma_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Normal'")
        normal_count = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(confidence) FROM predictions")
        avg_conf = cursor.fetchone()[0] or 0.0
    else:
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Glaucoma'")
        glaucoma_count = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Normal'")
        normal_count = cursor.fetchone()["count"]
        cursor.execute("SELECT AVG(confidence) FROM predictions")
        avg_conf = cursor.fetchone()["avg"] or 0.0

    conn.close()
    return {
        "total_screenings": total,
        "glaucoma_suspects": glaucoma_count,
        "normal_cases": normal_count,
        "average_confidence": round(float(avg_conf), 4)
    }

# Initialize on import
init_db()
