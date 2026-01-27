import os
import re
from typing import Dict, List, Tuple

# Optional PDF support
try:
    import PyPDF2
except Exception:
    PyPDF2 = None


# -----------------------------
# Local in-memory index
# -----------------------------
LOCAL_INDEX: Dict[str, List[Dict]] = {}
# Structure:
# LOCAL_INDEX[user_id] = [
#   {"source": "filename#chunk3", "sentences": [...], "tokens": set([...])}
# ]


# -----------------------------
# Text cleaning & utilities
# -----------------------------
def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _remove_pdf_noise(line: str) -> bool:
    """Return True if line looks like PDF junk."""
    line = line.strip().lower()
    if not line:
        return True
    if re.match(r"^\d+$", line):
        return True
    if "contents" in line or "foreword" in line:
        return True
    if len(line.split()) < 5:
        return True
    return False


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        if _remove_pdf_noise(s):
            continue
        if not re.match(r"^[A-Z]", s):
            continue
        cleaned.append(s)
    return cleaned


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _chunk_text(text: str, max_sentences: int = 5) -> List[List[str]]:
    sentences = _split_sentences(text)
    chunks = []
    buf = []

    for s in sentences:
        buf.append(s)
        if len(buf) >= max_sentences:
            chunks.append(buf)
            buf = []

    if buf:
        chunks.append(buf)

    return chunks


# -----------------------------
# File reading
# -----------------------------
def _read_txt_or_md(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(filepath: str) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is not installed. Install it to read PDFs.")
    text_parts = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _read_docx(filepath: str) -> str:
    try:
        from docx import Document
        doc = Document(filepath)
        text_parts = []
        for para in doc.paragraphs:
            text_parts.append(para.text)
        return "\n".join(text_parts)
    except Exception:
        return ""


# -----------------------------
# Ingestion
# -----------------------------
def ingest_local_document(user_id: str, filepath: str) -> None:
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)

    if ext in [".txt", ".md"]:
        raw = _read_txt_or_md(filepath)
    elif ext == ".pdf":
        raw = _read_pdf(filepath)
    elif ext == ".docx":
        raw = _read_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    raw = _clean_text(raw)
    sentence_chunks = _chunk_text(raw)

    items = []
    for i, sentences in enumerate(sentence_chunks):
        text_block = " ".join(sentences)
        items.append({
            "source": f"{filename}#chunk{i+1}",
            "sentences": sentences,
            "tokens": _tokenize(text_block)
        })

    LOCAL_INDEX.setdefault(user_id, [])
    LOCAL_INDEX[user_id].extend(items)


# -----------------------------
# Retrieval
# -----------------------------
def _retrieve_local(query: str, user_id: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
    q_tokens = _tokenize(query)
    candidates = LOCAL_INDEX.get(user_id, []) + LOCAL_INDEX.get("guest", [])

    scored = []
    for item in candidates:
        overlap = len(q_tokens & item["tokens"])
        score = overlap / max(len(q_tokens), 1)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# -----------------------------
# Confidence
# -----------------------------
def _confidence_from_results(scores: List[float]) -> Dict:
    if not scores:
        return {"score": 0.0, "label": "Low"}

    avg = sum(scores) / len(scores)

    if avg >= 0.75:
        label = "High"
    elif avg >= 0.4:
        label = "Medium"
    else:
        label = "Low"

    return {"score": round(avg, 2), "label": label}


# -----------------------------
# Main RAG
# -----------------------------
def run_rag_query(query: str, user_id: str) -> Dict:
    # Check if user has any uploaded documents
    user_docs = LOCAL_INDEX.get(user_id, []) + LOCAL_INDEX.get("guest", [])
    if not user_docs:
        return {
            "answer": "Please upload a document first, then ask your questions.",
            "confidence": {"score": 0.0, "label": "Low"},
            "sources": [],
        }

    results = _retrieve_local(query, user_id=user_id, top_k=5)

    if not results:
        return {
            "answer": "I couldn’t find relevant information in the uploaded documents.",
            "confidence": {"score": 0.0, "label": "Low"},
            "sources": [],
        }

    scores = []
    collected_sentences = []
    sources = []

    for score, item in results:
        scores.append(score)
        sources.append(item["source"])
        for s in item["sentences"]:
            if s not in collected_sentences:
                collected_sentences.append(s)

    # Keep answer focused (no dumping)
    final_sentences = collected_sentences[:6]

    answer = (
        "Based on the uploaded document content, here is a summary answer:\n\n"
        + " ".join(final_sentences)
    )

    return {
        "answer": answer,
        "confidence": _confidence_from_results(scores),
        "sources": sources,
    }
