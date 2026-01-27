import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# -----------------------------
# Paths + env loading (reliable)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)  # always loads C:\FYP\.env if app.py is in C:\FYP

DB_PATH = os.path.join(BASE_DIR, "database.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

# Limit uploads (adjust as needed)
MAX_UPLOAD_MB = 15

# Import AFTER loading env
from FYP_RAG.rag_query_ibm import run_rag_query, ingest_local_document


# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# Flask max request size (file upload)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# -----------------------------
# DB helpers
# -----------------------------
def init_db():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_history(user_id: str, question: str, answer: str):
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            "INSERT INTO history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (user_id, question, answer, datetime.utcnow().isoformat()),
        )


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


init_db()


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    # Quick backend check
    return jsonify(
        {
            "ok": True,
            "uploads_dir": os.path.exists(UPLOAD_DIR),
            "db_path": DB_PATH,
        }
    )


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    user_id = email if email else "guest"
    return jsonify({"user_id": user_id})


@app.route("/upload_docs", methods=["POST"])
def upload_docs():
    print("Upload request received")
    if "file" not in request.files:
        print("No file in request.files")
        return jsonify({"success": False, "message": "No file uploaded."}), 400

    file = request.files["file"]
    user_id = (request.form.get("user_id") or "guest").strip() or "guest"
    print(f"User ID: {user_id}, File: {file.filename}")

    if not file or not file.filename:
        print("Empty file or filename")
        return jsonify({"success": False, "message": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    print(f"Secure filename: {filename}")
    if not filename:
        print("Invalid filename after secure")
        return jsonify({"success": False, "message": "Invalid filename."}), 400

    if not allowed_file(filename):
        print(f"File not allowed: {filename}")
        return jsonify(
            {"success": False, "message": "Only .txt, .md, .pdf, .docx files are allowed."}
        ), 400

    # Save into uploads/<user_id>/
    safe_user = user_id.replace("\\", "_").replace("/", "_")
    user_dir = os.path.join(UPLOAD_DIR, safe_user)
    os.makedirs(user_dir, exist_ok=True)

    save_path = os.path.join(user_dir, filename)
    print(f"Saving to: {save_path}")
    file.save(save_path)

    # Index locally (works now). Later: swap to Discovery ingestion.
    try:
        print("Starting ingestion")
        ingest_local_document(user_id=user_id, filepath=save_path)
        print("Ingestion successful")
    except Exception as e:
        print(f"Ingestion error: {e}")
        return jsonify(
            {"success": False, "message": f"Uploaded but failed to index: {e}"}
        ), 500

    return jsonify(
        {
            "success": True,
            "message": f"Uploaded & indexed: {filename}",
            "path": save_path,
        }
    )


@app.route("/query_rag", methods=["POST"])
def query_rag():
    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or "guest").strip() or "guest"
    user_query = (data.get("query") or "").strip()

    if not user_query:
        return jsonify({"success": False, "answer": "Please enter a question."}), 400

    try:
        out = run_rag_query(query=user_query, user_id=user_id)

        answer = (out.get("answer") or "").strip()
        confidence = out.get("confidence") or {"score": 0.0, "label": "Low"}
        sources = out.get("sources") or []

        save_history(user_id, user_query, answer)

        return jsonify(
            {
                "success": True,
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "answer": f"Server error: {e}"}), 500


@app.route("/history", methods=["GET"])
def history():
    user_id = (request.args.get("user_id") or "guest").strip() or "guest"
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            """
            SELECT question, answer, created_at
            FROM history
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 50
            """,
            (user_id,),
        ).fetchall()

    items = [{"question": q, "answer": a, "created_at": t} for (q, a, t) in rows]
    return jsonify({"success": True, "user_id": user_id, "items": items})


# Better message when user uploads too large file
@app.errorhandler(413)
def too_large(_e):
    return (
        jsonify(
            {
                "success": False,
                "message": f"File too large. Max is {MAX_UPLOAD_MB}MB.",
            }
        ),
        413,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
