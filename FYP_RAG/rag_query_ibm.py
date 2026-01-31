# -----------------------------
# DOCX ingestion (python-docx fallback)
# -----------------------------
def ingest_docx_document(user_id: str, filepath: str):
    """
    Ingest a DOCX file using python-docx (no Docling dependency).
    Extracts paragraphs, chunks them, and stores in LOCAL_INDEX and Chroma.
    """
    import os
    import re
    import hashlib
    from docx import Document
    filename = os.path.basename(filepath)
    chunks = []
    try:
        # Compute a short hash of the file content for unique chunk IDs
        with open(filepath, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.sha1(file_bytes).hexdigest()[:8]
        doc = Document(filepath)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = " ".join(paragraphs)
        sentences = re.split(r"(?<=[.!?])\s+", full_text)
        title = None
        for p in paragraphs:
            if 5 <= len(p) <= 160:
                title = p
                break
        if not title and sentences:
            first_line = sentences[0].strip()
            if 5 <= len(first_line) <= 160:
                title = first_line
        meta = DOC_META.setdefault(user_id, {}).setdefault(filename, {})
        meta["title"] = title or ""
        for i in range(0, len(sentences), 4):
            block = " ".join(sentences[i:i+4]).strip()
            if not block:
                continue
            page_num = (i // 40) + 1
            chunks.append({
                "source": filename,
                "page": page_num,
                "chunk": ((i % 40) // 4) + 1,
                "text": block,
                "tokens": tokenize(block),
            })
        LOCAL_INDEX.setdefault(user_id, []).extend(chunks)
        try:
            col = get_chroma_collection(user_id)
            ids = []
            docs = []
            metas = []
            for c in chunks:
                # Add file_hash to chunk ID for uniqueness
                ids.append(f"{filename}_{file_hash}_p{c['page']}_c{c['chunk']}")
                docs.append(c["text"])
                metas.append({"source": filename, "page": c["page"], "chunk": c["chunk"]})
            # Delete any existing embeddings with these IDs before adding
            if ids:
                try:
                    col.delete(ids=ids)
                except Exception as del_e:
                    print(f"⚠️ Chroma delete failed for DOCX IDs: {del_e}")
            if docs:
                col.add(ids=ids, documents=docs, metadatas=metas)
        except Exception as e:
            print("⚠️ Chroma ingest failed for DOCX (LOCAL_INDEX only):", e)
    except Exception as e:
        print(f"❌ Failed to ingest DOCX: {filepath} — {e}")
        raise
    
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
# Document-level metadata cache: user_id -> filename -> {title: str | None}
DOC_META: Dict[str, Dict[str, Dict[str, str]]] = {}
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
        # Extract title/author/date from PDF metadata or first page
        title = None
        author = None
        pubdate = None
        try:
            info = getattr(reader, "metadata", None) or getattr(reader, "documentInfo", None)
            if info:
                t = getattr(info, "title", None) or info.get("/Title") if hasattr(info, "get") else None
                if t and isinstance(t, str) and len(t.strip()) >= 5:
                    title = t.strip()
                a = getattr(info, "author", None) or info.get("/Author") if hasattr(info, "get") else None
                if a and isinstance(a, str) and len(a.strip()) >= 2:
                    author = a.strip()
                # PDF dates often look like D:YYYYMMDDHHmmSSZ
                d = getattr(info, "creation_date", None) or info.get("/CreationDate") if hasattr(info, "get") else None
                d = d or (getattr(info, "mod_date", None) or info.get("/ModDate") if hasattr(info, "get") else None)
                if d and isinstance(d, str):
                    pubdate = _parse_pdf_date(d)
        except Exception:
            pass
        if title is None:
            try:
                first_text = clean_text(reader.pages[0].extract_text() or "") if reader.pages else ""
                first_line = re.split(r"\n|\r|(?<=[.!?])\s+", first_text)[0].strip()
                if 5 <= len(first_line) <= 160:
                    title = first_line
            except Exception:
                pass
        meta = DOC_META.setdefault(user_id, {}).setdefault(filename, {})
        meta["title"] = title or ""
        if author:
            meta["author"] = author
        if pubdate:
            meta["date"] = pubdate
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
    Prefer Docling + LangChain cookbook pipeline if available;
    fallback to PyPDF2-based ingestion otherwise.
    """
    try:
        ext = os.path.splitext(filepath)[-1].lower().strip('.')
        if docling is None:
            # If not PDF, we cannot parse without Docling
            if ext != "pdf":
                raise RuntimeError("Docling not available for non-PDF formats (DOCX/PPTX).")
            # For PDF, fallback to PyPDF2 ingestion
            return ingest_local_document(user_id, filepath)

        # Attempt cookbook pipeline imports inside try
        from langchain_community.document_loaders import DoclingLoader  # type: ignore
        from docling.document_converter import DocumentConverter  # type: ignore
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        from langchain_community.vectorstores import Chroma  # type: ignore

        converter = DocumentConverter()
        loader = DoclingLoader([filepath], converter=converter)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Normalize metadata to ensure consistent filtering by filename
        filename = os.path.basename(filepath)
        # Try to populate document-level title/author/date
        doc_title = None
        doc_author = None
        doc_date = None
        try:
            # Some loaders include title in metadata
            for d in documents:
                if hasattr(d, "metadata") and d.metadata:
                    dt = d.metadata.get("title") or d.metadata.get("document_title")
                    if dt and isinstance(dt, str) and len(dt.strip()) >= 5:
                        doc_title = dt.strip()
                    da = d.metadata.get("author") or d.metadata.get("authors")
                    if da and isinstance(da, str) and len(da.strip()) >= 2:
                        doc_author = da.strip()
                    dd = d.metadata.get("date") or d.metadata.get("publication_date") or d.metadata.get("created")
                    if dd and isinstance(dd, str) and len(dd.strip()) >= 4:
                        doc_date = dd.strip()
        except Exception:
            pass
        if doc_title is None:
            # Fallback to PDF inspection like local ingest
            try:
                with open(filepath, "rb") as _f:
                    r2 = PyPDF2.PdfReader(_f)
                    info = getattr(r2, "metadata", None) or getattr(r2, "documentInfo", None)
                    if info:
                        t = getattr(info, "title", None) or info.get("/Title") if hasattr(info, "get") else None
                        if t and isinstance(t, str) and len(t.strip()) >= 5:
                            doc_title = t.strip()
                        a = getattr(info, "author", None) or info.get("/Author") if hasattr(info, "get") else None
                        if a and isinstance(a, str) and len(a.strip()) >= 2:
                            doc_author = a.strip()
                        d = getattr(info, "creation_date", None) or info.get("/CreationDate") if hasattr(info, "get") else None
                        d = d or (getattr(info, "mod_date", None) or info.get("/ModDate") if hasattr(info, "get") else None)
                        if d and isinstance(d, str):
                            parsed = _parse_pdf_date(d)
                            if parsed:
                                doc_date = parsed
                    if doc_title is None and r2.pages:
                        first_text = clean_text(r2.pages[0].extract_text() or "")
                        first_line = re.split(r"\n|\r|(?<=[.!?])\s+", first_text)[0].strip()
                        if 5 <= len(first_line) <= 160:
                            doc_title = first_line
            except Exception:
                pass
        meta2 = DOC_META.setdefault(user_id, {}).setdefault(filename, {})
        meta2["title"] = doc_title or ""
        if doc_author:
            meta2["author"] = doc_author
        if doc_date:
            meta2["date"] = doc_date
        try:
            for d in splits:
                if not hasattr(d, "metadata") or d.metadata is None:
                    d.metadata = {}
                d.metadata["source"] = filename
                if "page" not in d.metadata:
                    d.metadata["page"] = d.metadata.get("page", 0)
        except Exception as _:
            pass

        # Persist to the same directory and collection naming used elsewhere
        vectorstore = Chroma.from_documents(
            splits,
            embeddings,
            collection_name=f"user_{user_id}",
            persist_directory=_get_vectorstore_path(),
        )
        try:
            vectorstore.persist()
        except Exception:
            pass

        # Also keep LOCAL_INDEX for fallback logic
        for i, d in enumerate(splits, start=1):
            text = clean_text(getattr(d, "page_content", ""))
            if not text:
                continue
            LOCAL_INDEX.setdefault(user_id, []).append({
                "source": filename,
                "page": d.metadata.get("page", 0) if hasattr(d, "metadata") else 0,
                "chunk": i,
                "text": text,
                "tokens": tokenize(text),
            })
    except Exception as e:
        # If PDF, we can fallback; else bubble up for graceful error
        if os.path.splitext(filepath)[-1].lower() == ".pdf":
            print("ℹ️ Cookbook ingestion unavailable, falling back to PyPDF2:", e)
            return ingest_local_document(user_id, filepath)
        raise



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
    # Stricter grounding: allow only 10% unknown tokens
    return (len(unknown) / max(len(a_tokens), 1)) <= 0.10


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
# Title query handling
# -----------------------------
def is_title_query(query: str) -> bool:
    q = query.lower().strip()
    return ("title" in q) and ("document" in q or "report" in q or "paper" in q or "pdf" in q)


def get_title_from_meta(user_id: str, source: str | None) -> str | None:
    if not source:
        return None
    return (DOC_META.get(user_id, {}).get(source, {}) or {}).get("title") or None


def is_author_query(query: str) -> bool:
    q = query.lower().strip()
    return ("author" in q or "written by" in q) and ("document" in q or "report" in q or "paper" in q or "pdf" in q)


def get_author_from_meta(user_id: str, source: str | None) -> str | None:
    if not source:
        return None
    return (DOC_META.get(user_id, {}).get(source, {}) or {}).get("author") or None


def is_pubdate_query(query: str) -> bool:
    q = query.lower().strip()
    return ("publication date" in q or "published" in q or "date" in q) and ("document" in q or "report" in q or "paper" in q or "pdf" in q)


def get_pubdate_from_meta(user_id: str, source: str | None) -> str | None:
    if not source:
        return None
    return (DOC_META.get(user_id, {}).get(source, {}) or {}).get("date") or None


def _parse_pdf_date(s: str) -> str | None:
    try:
        # PDF date format e.g., D:YYYYMMDDHHmmSSZ or with timezone offsets
        m = re.match(r"^D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?([+-Z].*)?$", s)
        if not m:
            # Try plain year or ISO-like
            y = re.match(r"^(\d{4})", s)
            return y.group(1) if y else s[:16]
        y, mo, d, hh, mm, ss = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
        parts = []
        if y:
            parts.append(y)
        if mo:
            parts.append(mo)
        if d:
            parts.append(d)
        date_str = "-".join(parts)
        return date_str or y
    except Exception:
        return None


# -----------------------------
# Summary handling
# -----------------------------
def is_summary_query(query: str) -> bool:
    q = query.lower().strip()
    return (
        "summary" in q
        or "summarize" in q
        or "key findings" in q
        or "overview" in q
        or "abstract" in q
    )


def extract_summary_from_doc(user_id: str, source: str | None) -> str | None:
    if not source:
        return None
    # Prefer first page content; else take top tokens from entire doc.
    doc_chunks = [d for d in LOCAL_INDEX.get(user_id, []) if d.get("source") == source]
    if not doc_chunks:
        return None
    first_page = min((d.get("page", 0) for d in doc_chunks if isinstance(d.get("page", 0), int)), default=0)
    fp_chunks = [d for d in doc_chunks if d.get("page", 0) == first_page]
    base_text = " ".join(clean_text(d.get("text", "")) for d in fp_chunks).strip()
    if not base_text:
        # Fallback: take first few chunks overall
        doc_chunks.sort(key=lambda x: (x.get("page", 0), x.get("chunk", 0)))
        base_text = " ".join(clean_text(d.get("text", "")) for d in doc_chunks[:5]).strip()
    if not base_text:
        return None
    # Keep a concise 3-5 sentence summary by truncation.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", base_text) if s.strip()]
    if not sentences:
        return None
    return " ".join(sentences[:5])


# -----------------------------
# RAG query (MAIN)
# -----------------------------
def run_rag_query(query: str, user_id: str, restrict_source: str | None = None):
    # If asking for document title, use metadata-based extraction
    if is_title_query(query):
        title = get_title_from_meta(user_id, restrict_source)
        if title:
            src_label = f"{restrict_source} — Page 1" if restrict_source else (restrict_source or "")
            return {
                "answer": title,
                "confidence": "High",
                "sources": [s for s in [src_label] if s],
                "retrieval": "doc-metadata",
            }
    # Author
    if is_author_query(query):
        author = get_author_from_meta(user_id, restrict_source)
        if author:
            src_label = f"{restrict_source} — Metadata" if restrict_source else (restrict_source or "")
            return {
                "answer": author,
                "confidence": "High",
                "sources": [s for s in [src_label] if s],
                "retrieval": "doc-metadata",
            }
    # Publication date
    if is_pubdate_query(query):
        pubdate = get_pubdate_from_meta(user_id, restrict_source)
        if pubdate:
            src_label = f"{restrict_source} — Metadata" if restrict_source else (restrict_source or "")
            return {
                "answer": pubdate,
                "confidence": "High",
                "sources": [s for s in [src_label] if s],
                "retrieval": "doc-metadata",
            }
    # Document summary
    if is_summary_query(query):
        summary = extract_summary_from_doc(user_id, restrict_source)
        if summary:
            src_label = f"{restrict_source} — Page 1" if restrict_source else (restrict_source or "")
            return {
                "answer": summary,
                "confidence": "High",
                "sources": [s for s in [src_label] if s],
                "retrieval": "doc-extractive",
            }
    # Optional: try Granite Snack Cookbook pattern via LangChain
    try:
        if (os.getenv("RAG_PIPELINE_MODE", "fallback").strip().lower() == "cookbook"):
            from langchain.chains import RetrievalQA  # type: ignore
            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain_community.llms import IBMWatsonxAI  # type: ignore
            from langchain_community.vectorstores import Chroma  # type: ignore
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

            # Load existing persistent vectorstore for this user
            model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            vectorstore = Chroma(
                collection_name=f"user_{user_id}",
                embedding_function=embeddings,
                persist_directory=_get_vectorstore_path(),
            )

            granite_llm = IBMWatsonxAI(
                url=os.getenv("WATSONX_URL"),
                apikey=os.getenv("WATSONX_API_KEY"),
                project_id=os.getenv("IBM_PROJECT_ID"),
                model_id=os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-8b-instruct"),
                params={
                    "temperature": 0.1,
                    "max_new_tokens": 512,
                    "repetition_penalty": 1.1,
                },
            )

            prompt_template = (
                "Use the following context to answer the question. "
                "If you cannot find the answer in the context, say \"Insufficient information in provided context.\"\n\n"
                "Context: {context}\n\nQuestion: {question}\n\nAnswer: "
            )
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            # Support source restriction via Chroma filter
            search_kwargs = {"k": 4}
            if restrict_source:
                search_kwargs["filter"] = {"source": restrict_source}

            qa_chain = RetrievalQA.from_chain_type(
                llm=granite_llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs=search_kwargs),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
            )

            result = qa_chain({"query": query})
            answer = result.get("result", "")
            sources_docs = result.get("source_documents", [])
            sources_list = []
            for d in sources_docs[:3]:
                src = d.metadata.get("source") if hasattr(d, "metadata") else None
                page = d.metadata.get("page") if hasattr(d, "metadata") else None
                if src is None:
                    continue
                label = f"{src} — Page {page}" if page is not None else src
                if label not in sources_list:
                    sources_list.append(label)
            conf = "High" if len(sources_docs) >= 3 else ("Medium" if len(sources_docs) >= 1 else "Low")
            return {
                "answer": answer or "Insufficient information in provided context.",
                "confidence": f"{conf}",
                "sources": sources_list,
                "retrieval": "langchain-retrievalqa",
            }
    except Exception as e:
        print("ℹ️ LangChain cookbook path unavailable, using fallback:", e)

    q_tokens = tokenize(query)

    top = []
    context = ""
    retrieval_method = "fallback-token"
    # First try: Chroma similarity search
    avg_sim = 0.0
    token_best = 0.0
    try:
        col = get_chroma_collection(user_id)
        where = {"source": restrict_source} if restrict_source else None
        results = col.query(
            query_texts=[query],
            n_results=4,
            include=["documents", "metadatas", "distances"],
            where=where,
        )
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
        if restrict_source:
            docs = [d for d in docs if d.get("source") == restrict_source]
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

    # If answer is generic/hallucinated, or not grounded, fallback to extractive
    hallucination_phrases = [
        "based on the document", "as stated above", "according to the document", "as mentioned earlier",
        "the document states", "the document mentions", "as per the document", "as described above"
    ]
    is_hallucinated = any(p in answer.lower() for p in hallucination_phrases)
    if (
        answer.strip().lower() == "insufficient information in provided context."
        or not grounding_gate(answer, context, query)
        or is_hallucinated
    ):
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
