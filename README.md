# FYP Technology Stack & Configuration

## Stack Overview
- Python for core development.
- Flask for web-based UI/API (see `app.py`).
- LangChain runnables for simple RAG orchestration (see `FYP_RAG/rag_pipeline_langchain.py`).
- IBM Granite 3.x via watsonx for answer generation (see `FYP_RAG/rag_query_ibm.py`).
- Docling preferred for document parsing, with PyPDF2 fallback.
- Chroma persistent vector store (see `vectorstore/` contents).
- Embeddings: configurable — Granite (watsonx) or local Sentence-Transformers.

## Embedding Backend Toggle
You can choose the embedding backend used by Chroma:

- `EMBEDDING_BACKEND=granite` — uses IBM watsonx Granite embeddings.
- `EMBEDDING_BACKEND=sentence-transformers` — uses a local model (`all-MiniLM-L6-v2` by default).

When misconfigured or unavailable, the system safely falls back to `sentence-transformers` without breaking ingestion or queries.

## RAG Pipeline Mode
You can opt into the IBM Granite Snack Cookbook-aligned pipeline while preserving fallback:

- Set `RAG_PIPELINE_MODE=cookbook` to use Docling (`DocumentConverter` + `DoclingLoader`), LangChain `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, and LangChain `Chroma` + `RetrievalQA` with Granite.
- Omit or set `RAG_PIPELINE_MODE=fallback` to use the original custom pipeline (PyPDF2 fallback, direct Chroma client, REST call to Granite, grounding + extractive fallback).

If any cookbook component fails or is unavailable, the system automatically switches to the fallback path to avoid breaking queries.

### Required Environment Variables
For Granite embeddings:
- `WATSONX_URL` — Base URL of your watsonx instance.
- `IBM_PROJECT_ID` — watsonx project ID.
- `WATSONX_API_KEY` — API key for authentication.
- `GRANITE_EMBEDDING_MODEL_ID` — e.g., `ibm/granite-embedding-...` (consult IBM console).

For Sentence-Transformers (optional override):
- `SENTENCE_TRANSFORMER_MODEL` — e.g., `sentence-transformers/all-MiniLM-L6-v2`.

Set these in your environment or a `.env` file at repo root.

## How It Works
- The embedding selection is implemented in `get_chroma_collection()` inside `FYP_RAG/rag_query_ibm.py`.
- With `granite` selected and valid env, `WatsonxGraniteEmbeddingFunction` calls the `ml/v1/text/embeddings` REST endpoint and feeds vectors to Chroma.
- On any init/call failure or missing env, the code logs a warning and uses Sentence-Transformers instead.

## Quick Checks
- Verify Granite model availability:
  ```bash
  python FYP_RAG/check_watsonx_embeddings.py
  ```
- Run the app (Flask):
  ```bash
  python app.py
  ```

## Notes
- Answer generation continues to use Granite `ibm/granite-3-8b-instruct` via REST.
- Vector store paths remain under `vectorstore/`; no DB migrations required.
- No breaking changes: existing ingestion and query flows continue to work with fallback behavior.
