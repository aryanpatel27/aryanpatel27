"""
app.py - Flask REST API for the AuroraStay policy assistant.

Heavy objects (embedding model, FAISS index, local LLM when no API key is set)
are loaded ONCE at import time so every request only pays for retrieval and
generation.

Endpoints (all JSON):
    GET  /                                  health message
    GET  /health                            {"status": "ok", ...}
    POST /v1/relevant_chunks                {query, k}                       -> top-k chunks
    POST /v1/answer                         {query, k, model_name, temperature, top_p, max_tokens} -> answer only
    POST /v1/answer_with_relevant_chunks    same input                        -> answer + chunks
"""

import math
import os
import time

from flask import Flask, jsonify, request

import rag_utils as R

INDEX_DIR = os.getenv("FAISS_INDEX_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index"))
MAX_QUERY_CHARS = 2000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024   # reject oversized request bodies (the API is on a public URL)

# ---- one-time start-up ----------------------------------------------------- #
_t0 = time.time()
print(f"[startup] loading embedding model {R.EMBEDDING_MODEL_NAME} ...", flush=True)
EMBEDDINGS = R.get_embeddings(R.EMBEDDING_MODEL_NAME)
print(f"[startup] loading FAISS index from {INDEX_DIR} ...", flush=True)
VECTORSTORE = R.load_vectorstore(INDEX_DIR, EMBEDDINGS)
print(f"[startup] LLM provider: {R.llm_provider()} ({R.active_llm_name()})", flush=True)
if R.llm_provider() == "local":
    R.get_local_llm()  # warm the local model so the first request is not slow
print(f"[startup] ready in {time.time() - _t0:.1f}s - {VECTORSTORE.index.ntotal} chunks indexed", flush=True)


# ---- helpers --------------------------------------------------------------- #
def _params(payload: dict) -> dict:
    """Validate / default the generation parameters sent by the frontend."""
    try:
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        if not (math.isfinite(temperature) and 0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0 and 2")
        if not (math.isfinite(top_p) and 0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        return {
            "k": max(1, min(int(payload.get("k", 4)), 10)),
            "model_name": str(payload.get("model_name") or R.DEFAULT_LLM_NAME),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max(32, min(int(payload.get("max_tokens", 512)), 2048)),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid parameter: {exc}") from exc


def _query(payload) -> str:
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    q = payload.get("query", "")
    if not isinstance(q, str) or not q.strip():
        raise ValueError("'query' must be a non-empty string")
    if len(q) > MAX_QUERY_CHARS:
        raise ValueError(f"'query' is too long (max {MAX_QUERY_CHARS} characters)")
    return q.strip()


def _error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ---- routes ---------------------------------------------------------------- #
@app.get("/")
def index():
    return "Welcome to the Aurora Stay RAG API!"


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "embedding_model": R.EMBEDDING_MODEL_NAME,
        "llm_provider": R.llm_provider(),
        "llm": R.active_llm_name(),
        "chunks_indexed": int(VECTORSTORE.index.ntotal),
    })


@app.post("/v1/relevant_chunks")
def relevant_chunks():
    payload = request.get_json(silent=True)
    try:
        query = _query(payload)
        k = _params(payload)["k"]
    except ValueError as exc:
        return _error(str(exc))
    chunks = R.retrieve(query, k, VECTORSTORE)
    return jsonify({"query": query, "k": k, "relevant_chunks": chunks})


@app.post("/v1/answer")
def answer():
    payload = request.get_json(silent=True)
    try:
        query = _query(payload)
        p = _params(payload)
    except ValueError as exc:
        return _error(str(exc))
    try:
        result = R.rag(query, vectorstore=VECTORSTORE, **p)
    except Exception as exc:  # surface LLM/provider failures as JSON, not HTML 500s
        return _error(f"generation failed: {exc}", 500)
    result.pop("relevant_chunks", None)
    return jsonify(result)


@app.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    payload = request.get_json(silent=True)
    try:
        query = _query(payload)
        p = _params(payload)
    except ValueError as exc:
        return _error(str(exc))
    try:
        result = R.rag(query, vectorstore=VECTORSTORE, **p)
    except Exception as exc:
        return _error(f"generation failed: {exc}", 500)
    return jsonify(result)


if __name__ == "__main__":  # local debugging only; the container uses gunicorn
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")), debug=False)
