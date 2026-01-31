import os
import math
import re
import requests
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2
import chromadb
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions
try:
    import docling
except Exception:
    docling = None

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

print("✅ WATSONX_API_KEY loaded:", bool(os.getenv("WATSONX_API_KEY")))
print("✅ IBM_PROJECT_ID loaded:", bool(os.getenv("IBM_PROJECT_ID")))
print("✅ WATSONX_URL:", os.getenv("WATSONX_URL"))


# In-memory index: user_id -> chunks
LOCAL_INDEX: Dict[str, List[dict]] = {}
STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","that","this","those","these",
    "is","are","was","were","be","been","being","of","in","on","for","to","with","without",
    "by","as","at","from","it","its","their","there","here","such","can","may","might","should",
    "must","could","will","would","do","does","did","not","no","yes","about","into","within","between",
}


# -----------------------------
# Chroma setup + IBM Embeddings
# -----------------------------
def _get_vectorstore_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "vectorstore")


def get_chroma_collection(user_id: str):
    client = chromadb.PersistentClient(path=_get_vectorstore_path())
    # Use local sentence-transformers embeddings (offline-friendly)
    emb_model = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    emb_fn = None
    try:
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_model)
        print(f"✅ Using local embeddings: {emb_model}")
    except Exception as e:
        print("⚠️ SentenceTransformer embeddings unavailable; proceeding without embedding function:", e)
    return client.get_or_create_collection(
        name=f"user_{user_id}",
        metadata={"hnsw:space": "cosine"},
        embedding_function=emb_fn,
    )


# -----------------------------
# Utils
# -----------------------------
def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


# -----------------------------
# Document ingestion
# -----------------------------
def ingest_local_document(user_id: str, filepath: str):
    filename = os.path.basename(filepath)
    chunks = []

    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if not text:
                continue

            sentences = re.split(r"(?<=[.!?])\s+", text)
            for i in range(0, len(sentences), 4):
                block = " ".join(sentences[i:i + 4])
                if not block.strip():
                    continue

                chunks.append({
                    "source": filename,
                    "page": page_num,
                    "chunk": (i // 4) + 1,
                    "text": block,
                    "tokens": tokenize(block),
                })

    LOCAL_INDEX.setdefault(user_id, []).extend(chunks)

    # Also add to Chroma for vector retrieval (best-effort; fall back if embeddings unavailable)
    try:
        col = get_chroma_collection(user_id)
        ids = []
        docs = []
        metas = []
        for c in chunks:
            ids.append(f"{filename}_p{c['page']}_c{c['chunk']}")
            docs.append(c["text"])
            metas.append({"source": filename, "page": c["page"], "chunk": c["chunk"]})
        if docs:
            col.add(ids=ids, documents=docs, metadatas=metas)
    except Exception as e:
        print("⚠️ Chroma ingest failed (falling back to LOCAL_INDEX only):", e)


# -----------------------------
# Docling ingestion (best-effort)
# -----------------------------
def ingest_document_docling(user_id: str, filepath: str):
    """
    Prefer Docling for robust parsing + chunking if available;
    fallback to PyPDF2-based ingestion otherwise.
    """
    if docling is None:
        print("ℹ️ Docling not available, using PyPDF2 ingestion.")
        return ingest_local_document(user_id, filepath)

    filename = os.path.basename(filepath)
    chunks = []
    try:
        # Docling API varies; attempt generic pipeline and fallback on error
        # Use docling.parse to extract text segments, if supported
        from docling_parse import Parser  # type: ignore
        parser = Parser()
        doc = parser.parse(filepath)
        texts = []
        try:
            # Collect paragraphs or segments (best-effort)
            texts = [seg.text for seg in getattr(doc, "segments", []) if getattr(seg, "text", "")]  # type: ignore
        except Exception:
            pass
        if not texts:
            # Fallback: use PyPDF2 path
            print("ℹ️ Docling parse returned no segments, falling back to PyPDF2.")
            return ingest_local_document(user_id, filepath)

        # Group texts into blocks of ~4 segments
        for i in range(0, len(texts), 4):
            block = clean_text(" ".join(texts[i:i+4]))
            if not block:
                continue
            chunks.append({
                "source": filename,
                "page": 0,
                "chunk": (i // 4) + 1,
                "text": block,
                "tokens": tokenize(block),
            })
    except Exception as e:
        print("⚠️ Docling ingestion failed, using PyPDF2:", e)
        return ingest_local_document(user_id, filepath)

    LOCAL_INDEX.setdefault(user_id, []).extend(chunks)

    # Sync into Chroma
    try:
        col = get_chroma_collection(user_id)
        ids = []
        docs = []
        metas = []
        for c in chunks:
            ids.append(f"{filename}_p{c['page']}_c{c['chunk']}")
            docs.append(c["text"])
            metas.append({"source": filename, "page": c["page"], "chunk": c["chunk"]})
        if docs:
            col.add(ids=ids, documents=docs, metadatas=metas)
    except Exception as e:
        print("⚠️ Chroma ingest failed (Docling path):", e)


# -----------------------------
# Watsonx / Granite
# -----------------------------
def get_iam_token():
    api_key = os.getenv("WATSONX_API_KEY")
    if not api_key:
        raise RuntimeError("Missing WATSONX_API_KEY")

    res = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "apikey": api_key,
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        },
        timeout=15
    )

    if res.status_code != 200:
        raise RuntimeError(f"IAM token failed: {res.text}")

    return res.json().get("access_token")


def call_granite(question: str, context: str) -> str:
    token = get_iam_token()

    url = os.getenv("WATSONX_URL")
    project_id = os.getenv("IBM_PROJECT_ID")
    if not url or not project_id:
        raise RuntimeError("Watsonx env vars missing")

    endpoint = f"{url}/ml/v1/text/chat?version=2024-02-15"

    payload = {
        "model_id": "ibm/granite-3-8b-instruct",
        "project_id": project_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Use only the provided context. "
                    "Answer the question in 1–2 concise sentences. "
                    "If the context is insufficient, reply exactly: 'Insufficient information in provided context.' "
                    "Do not add background or speculation."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        "parameters": {
            "temperature": 0.0,
            "top_p": 0.1,
            "max_new_tokens": 160
        }
    }

    res = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )

    # Handle quota / auth issues
    if res.status_code in (401, 403, 429):
        raise RuntimeError(f"Watsonx quota/auth error: {res.status_code}")

    if res.status_code != 200:
        raise RuntimeError(f"Watsonx error {res.status_code}: {res.text}")

    try:
        data = res.json()
    except Exception:
        raise RuntimeError(f"Non-JSON Watsonx response: {res.text}")

    # Support both schemas
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()

    if "results" in data:
        return data["results"][0]["generated_text"].strip()

    raise RuntimeError(f"Unexpected Watsonx response: {data}")


# -----------------------------
# Numeric policy: round decimals up, no fractional output
# -----------------------------
def normalize_numbers(text: str) -> str:
    """
    Convert decimal numbers to integers by rounding up (ceiling),
    preserving any immediate unit (e.g., %, kg). Avoid touching
    integers or dates/years. Best-effort, only transforms tokens
    containing a decimal point.
    """
    if not text:
        return text

    def repl(m):
        num = m.group(1)
        unit = m.group(2) or ""
        try:
            val = float(num)
            rounded = math.ceil(val)
            return f"{rounded}{unit}"
        except Exception:
            return m.group(0)

    # Match decimals like 3.2, -0.7, 5.0 optionally followed by a unit or %
    pattern = r"(?<![\w/])(-?\d+\.\d+)(\s*(?:%|percent|million|billion|thousand|kg|g|mg|km|m|cm|mm|hours|days|weeks|months|years)?)"
    return re.sub(pattern, repl, text)


# -----------------------------
# Grounding gate
# -----------------------------
def grounding_gate(answer: str, context: str, query: str) -> bool:
    """
    Returns True if the answer appears grounded in the provided context
    and query (i.e., uses mostly tokens present in them), otherwise False.
    """
    a_tokens = tokenize(answer) - STOPWORDS
    c_tokens = tokenize(context)
    q_tokens = tokenize(query)
    allowed = c_tokens | q_tokens

    if not a_tokens:
        return True

    unknown = a_tokens - allowed
    # Simpler grounding: allow small amount of connective wording
    return (len(unknown) / max(len(a_tokens), 1)) <= 0.30


# -----------------------------
# Extractive fallback
# -----------------------------
def extractive_fallback(top, q_tokens):
    sentences = []
    for _, d in top:
        txt = d.get("text") or ""
        if txt:
            sentences.extend(re.split(r"(?<=[.!?])\s+", txt))

    if not sentences:
        return top[0][1].get("text", "").strip()

    scored = []
    for s in sentences:
        overlap = len(tokenize(s) & q_tokens)
        if overlap > 0:
            scored.append((overlap, s))

    if not scored:
        return sentences[0].strip()

    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join(s for _, s in scored[:4]).strip()


# -----------------------------
# Extractive cause/reason miner
# -----------------------------
CAUSE_HINTS = {
    "because", "due to", "owing to", "driven by", "led to", "lead to",
    "attributed to", "resulted in", "results from", "contributors", "drivers",
    "factors", "main reasons", "reason for", "decline in", "reduction in",
    "improvements in", "expansion of", "increase in coverage", "vaccination",
    "immunization", "primary care", "sanitation", "water", "nutrition",
    "maternal", "child mortality", "infectious diseases"
}


def is_cause_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in [
        "why ", "reason", "reasons", "cause", "causes", "drivers", "contributors", "factors",
        "what were the main reasons", "what are the main reasons"
    ])


def extract_reasons_from_docs(top, q_tokens, max_items: int = 5):
    items = []  # list of (score, text, src)
    seen = set()
    for sim, d in top:
        txt = d.get("text") or ""
        if not txt:
            continue
        src = f"{d.get('source')} — Page {d.get('page')} (Chunk {d.get('chunk')})"
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            low = s_clean.lower()
            # Heuristics: has cause hint or contains overlap with query tokens
            if (any(h in low for h in CAUSE_HINTS) or len(tokenize(low) & q_tokens) >= max(2, int(0.2 * max(len(q_tokens), 1)))):
                # Prefer concise paraphrasable clauses; trim overly long sentences
                if len(s_clean) > 300:
                    continue
                key = re.sub(r"\W+", " ", low)[:120]
                if key in seen:
                    continue
                seen.add(key)
                # Score by retrieval sim and token overlap
                score = sim + 0.3 * (len(tokenize(low) & q_tokens) / max(len(q_tokens), 1))
                items.append((score, s_clean, src))
    if not items:
        return []
    items.sort(key=lambda x: x[0], reverse=True)
    # Deduplicate by similar starts
    out = []
    used_prefix = set()
    for _, text, src in items:
        pref = text[:80].lower()
        if any(pref in u or u in pref for u in used_prefix):
            continue
        used_prefix.add(pref)
        out.append((text, src))
        if len(out) >= max_items:
            break
    return out


# -----------------------------
# Query typing (lightweight)
# -----------------------------
def is_numeric_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in [
        "how many", "how much", "what percentage", "percent", "%", "rate", "ratio"
    ])


def is_list_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in [
        "list ", "what are the", "which are the", "key", "main", "types of", "categories of", "examples of"
    ])


def is_definition_query(query: str) -> bool:
    q = query.lower().strip()
    return q.startswith("what is ") or q.startswith("define ") or "meaning of" in q or "refers to" in q


# -----------------------------
# Extractors for numeric, list, definition
# -----------------------------
def extract_numeric_from_docs(query: str, top, q_tokens):
    items = []  # (score, phrase, src)
    for sim, d in top:
        txt = d.get("text") or ""
        if not txt:
            continue
        src = f"{d.get('source')} — Page {d.get('page')} (Chunk {d.get('chunk')})"
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        for s in sentences:
            if not re.search(r"\d", s):
                continue
            low = s.lower()
            if len(low) > 280:
                continue
            overlap = len(tokenize(low) & q_tokens)
            score = sim + 0.4 * (overlap / max(len(q_tokens), 1))
            items.append((score, s.strip(), src))
    if not items:
        return None, []
    items.sort(key=lambda x: x[0], reverse=True)
    best_text, best_src = items[0][1], items[0][2]
    return best_text, [best_src]


def extract_list_from_docs(top, q_tokens, max_items: int = 5):
    candidates = []  # (score, item, src)
    hint_words = {"include", "includes", "including", "such as", "comprise", "consist", "are", "were", "involve"}
    for sim, d in top:
        txt = d.get("text") or ""
        if not txt:
            continue
        src = f"{d.get('source')} — Page {d.get('page')} (Chunk {d.get('chunk')})"
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        for s in sentences:
            low = s.lower()
            if not any(h in low for h in hint_words):
                continue
            # Split enumerations by comma/semicolon
            parts = [p.strip() for p in re.split(r",|;", s) if p.strip()]
            # If splitting produced too many tiny parts, keep sentence-level
            if len(parts) <= 1:
                parts = [s.strip()]
            for p in parts:
                if 2 <= len(p) <= 120:
                    overlap = len(tokenize(p.lower()) & q_tokens)
                    score = sim + 0.3 * (overlap / max(len(q_tokens), 1))
                    candidates.append((score, p, src))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for _, item, src in candidates:
        key = re.sub(r"\W+", " ", item.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((item, src))
        if len(out) >= max_items:
            break
    return out


def extract_definition_from_docs(query: str, top, q_tokens):
    term = None
    m = re.match(r"\s*(what is|define)\s+(.+?)\?*$", query.strip(), flags=re.IGNORECASE)
    if m:
        term = m.group(2).strip()
    term_low = term.lower() if term else None
    best = None
    for sim, d in top:
        txt = d.get("text") or ""
        if not txt:
            continue
        src = f"{d.get('source')} — Page {d.get('page')} (Chunk {d.get('chunk')})"
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        for s in sentences:
            t = s.strip()
            if not t or len(t) > 240:
                continue
            low = t.lower()
            if term_low and term_low not in low:
                continue
            if re.search(r"\bis\b|\brefers to\b|\bdefined as\b", low):
                overlap = len(tokenize(low) & q_tokens)
                score = sim + 0.4 * (overlap / max(len(q_tokens), 1))
                if not best or score > best[0]:
                    best = (score, t, src)
    if best:
        return best[1], [best[2]]
    return None, []


# -----------------------------
# RAG query (MAIN)
# -----------------------------
def run_rag_query(query: str, user_id: str):
    q_tokens = tokenize(query)

    top = []
    context = ""
    retrieval_method = "fallback-token"
    # First try: Chroma similarity search
    avg_sim = 0.0
    token_best = 0.0
    try:
        col = get_chroma_collection(user_id)
        results = col.query(query_texts=[query], n_results=5, include=["documents", "metadatas", "distances"])
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        # Convert distances to similarity (cosine space)
        for doc, meta, dist in zip(docs, metas, dists):
            sim = 1.0 - float(dist)
            top.append((sim, {"text": doc, "source": meta.get("source"), "page": meta.get("page"), "chunk": meta.get("chunk")}))
        top.sort(key=lambda x: x[0], reverse=True)
        context = " ".join(d["text"] for _, d in top)[:6000]
        retrieval_method = "vector"
        if top:
            avg_sim = sum(s for s, _ in top) / len(top)
    except (ChromaError, Exception) as e:
        print("⚠️ Chroma query failed, falling back to token overlap:", e)
        # Fallback: token overlap over LOCAL_INDEX
        docs = LOCAL_INDEX.get(user_id, [])
        scored = []
        for d in docs:
            score = len(q_tokens & d["tokens"]) / max(len(q_tokens), 1)
            if score > 0.1:
                scored.append((score, d))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:5]
            context = " ".join(d["text"] for _, d in top)[:6000]
            token_best = scored[0][0]

    if not top:
        return {
            "answer": "Insufficient information in provided context.",
            "confidence": "Low (0.00)",
            "sources": [],
            "retrieval": retrieval_method,
        }

    # Simple synthesis: generate concise answer from context, else fallback
    try:
        answer = call_granite(query, context)
    except Exception as e:
        print("⚠️ Granite failed, using fallback:", e)
        answer = extractive_fallback(top, q_tokens)

    # Grounding: if insufficient or ungrounded, use extractive fallback
    if answer.strip().lower() == "insufficient information in provided context." or not grounding_gate(answer, context, query):
        answer = extractive_fallback(top, q_tokens)

    avg = sum(s for s, _ in top) / len(top)
    label = "High" if avg >= 0.6 else "Medium"

    # Build sources list simply from top retrieved chunks
    sources = []
    for _, d in top[:3]:
        src = f"{d['source']} — Page {d['page']} (Chunk {d['chunk']})"
        if src not in sources:
            sources.append(src)

    return {
        "answer": answer,
        "confidence": f"{label} ({avg:.2f})",
        "sources": sources,
        "retrieval": retrieval_method,
    }
