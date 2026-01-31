import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv
import time

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB cap

# Allowed file types for ingestion
ALLOWED_EXTENSIONS = {"pdf", "docx"}

# -----------------------------
# RAG engine
# -----------------------------
from FYP_RAG.rag_query_ibm import run_rag_query, ingest_document_docling, ingest_local_document, ingest_docx_document
from FYP_RAG.ibm_cos_storage import cos_enabled, upload_file_to_cos

# -----------------------------
# Paths & config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "database.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Database
# -----------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                query TEXT,
                answer TEXT,
                confidence TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS perf_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                kind TEXT,
                ref TEXT,
                duration_ms INTEGER,
                success INTEGER,
                timestamp TEXT
            )
        """)

def get_last_ingested_filename(user_id: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT ref FROM perf_logs
                WHERE user_id = ? AND kind = 'ingest' AND success = 1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if row:
                return row["ref"]
    except Exception:
        pass
    return None

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_docs", methods=["POST"])
def upload_docs():
    file = request.files.get("file")
    user_id = request.form.get("user_id", "guest")

    if not file or file.filename == "":
        return jsonify(success=False, message="No file uploaded"), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(success=False, message="Unsupported file type. Allowed: PDF, DOCX"), 400
    user_dir = os.path.join(UPLOAD_FOLDER, user_id)
    os.makedirs(user_dir, exist_ok=True)

    path = os.path.join(user_dir, filename)
    start = time.perf_counter()
    file.save(path)

    try:
        # Only PDF and DOCX supported. Use python-docx for DOCX, fallback to existing for PDF.
        if ext == "docx":
            ingest_docx_document(user_id, path)
        else:
            ingest_document_docling(user_id, path)
        # Best-effort: upload original to IBM Cloud Object Storage if configured
        cos_url = None
        if cos_enabled():
            try:
                key = f"{user_id}/{filename}"
                cos_url = upload_file_to_cos(path, key)
            except Exception as cos_err:
                # Log to console only; do not fail ingest if COS upload fails
                print("⚠️ COS upload failed:", cos_err)
        duration_ms = int((time.perf_counter() - start) * 1000)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO perf_logs (user_id, kind, ref, duration_ms, success, timestamp)
                VALUES (?, 'ingest', ?, ?, 1, ?)
                """,
                (user_id, filename, duration_ms, datetime.now().isoformat(timespec="seconds"))
            )
        return jsonify(success=True, message="Successfully uploaded and ingested document", duration_ms=duration_ms, cos_url=cos_url)
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO perf_logs (user_id, kind, ref, duration_ms, success, timestamp)
                VALUES (?, 'ingest', ?, ?, 0, ?)
                """,
                (user_id, filename, duration_ms, datetime.now().isoformat(timespec="seconds"))
            )
        return jsonify(success=False, message=f"Ingestion failed for {ext.upper()} file: {e}"), 500


@app.route("/query_rag", methods=["POST"])
def query_rag():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    user_id = data.get("user_id", "guest")
    current_only = data.get("current_only")
    if current_only is None:
        current_only = (os.getenv("RAG_CURRENT_ONLY", "false").strip().lower() == "true")

    if not query:
        return jsonify(success=False, answer="Empty query"), 400

    try:
        start = time.perf_counter()
        restrict_source = None
        if current_only:
            restrict_source = get_last_ingested_filename(user_id)
        result = run_rag_query(query, user_id, restrict_source=restrict_source)
        duration_ms = int((time.perf_counter() - start) * 1000)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO query_logs
                (user_id, query, answer, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    query,
                    result["answer"],
                    result["confidence"],
                    datetime.now().isoformat(timespec="seconds"),
                )
            )
            conn.execute(
                """
                INSERT INTO perf_logs (user_id, kind, ref, duration_ms, success, timestamp)
                VALUES (?, 'query', ?, ?, 1, ?)
                """,
                (user_id, query[:200], duration_ms, datetime.now().isoformat(timespec="seconds"))
            )

        return jsonify(
            success=True,
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            retrieval=result.get("retrieval"),
            restrict_source=restrict_source,
            duration_ms=duration_ms,
        )

    except Exception as e:
        try:
            duration_ms = int((time.perf_counter() - start) * 1000)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """
                    INSERT INTO perf_logs (user_id, kind, ref, duration_ms, success, timestamp)
                    VALUES (?, 'query', ?, ?, 0, ?)
                    """,
                    (user_id, query[:200], duration_ms, datetime.now().isoformat(timespec="seconds"))
                )
        except Exception:
            pass
        return jsonify(
            success=False,
            answer="Internal error during RAG synthesis",
            error=str(e),
        ), 500


@app.route("/history", methods=["GET"])
def history():
    user_id = request.args.get("user_id", "guest")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, query, answer, confidence, timestamp
            FROM query_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 100
            """,
            (user_id,),
        ).fetchall()

    return jsonify(
        success=True,
        history=[dict(row) for row in rows]
    )


@app.route("/history/delete", methods=["POST"])
def history_delete():
    data = request.get_json(silent=True) or {}
    item_id = data.get("id")
    user_id = data.get("user_id", "guest")

    if not item_id:
        return jsonify(success=False, message="Missing id"), 400

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "DELETE FROM query_logs WHERE id = ? AND user_id = ?",
            (item_id, user_id),
        )

    return jsonify(success=True, deleted=cur.rowcount)


@app.route("/history/clear", methods=["POST"])
def history_clear():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "guest")

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "DELETE FROM query_logs WHERE user_id = ?",
            (user_id,),
        )

    return jsonify(success=True, deleted=cur.rowcount)

# -----------------------------
# Main (CRITICAL FIX HERE)
# -----------------------------
if __name__ == "__main__":
    init_db()
    # Startup diagnostics for clarity
    rag_mode = os.getenv("RAG_PIPELINE_MODE", "fallback")
    emb_model = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    granite_model = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct")
    print("🔧 RAG mode:", rag_mode)
    print("🔧 Embedding model:", emb_model)
    print("🔧 Granite model:", granite_model)
    print("🔥 Flask running on port 5001")

    # 🚨 IMPORTANT:
    # use_reloader=False prevents Flask from running TWO processes
    # which was wiping your in-memory RAG index
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        use_reloader=False
    )

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify(success=False, message="File too large. Max 25 MB."), 413
