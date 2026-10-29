# AuroraStay Hotels – Policy Query Assistant (RAG POC)

A Retrieval-Augmented Generation proof of concept that lets AuroraStay hotel staff ask
natural-language questions and receive answers grounded **only** in the group's four
official policy documents, with the supporting source excerpts shown next to every answer.

| Folder / file | What it is |
|---|---|
| `AuroraStay_RAG_POC.ipynb` / `.html` | Full-code notebook: data understanding → chunking → embeddings + FAISS → grounded generation → DeepEval evaluation → deployment files → online inference → insights & recommendations. **The `.html` export is the submission artefact.** |
| `data/` | The four policy PDFs plus `evaluation.csv` (10 benchmark queries) and `test.xlsx` (10 test queries). |
| `backend/` | Flask REST API (`app.py`), the RAG pipeline (`rag_utils.py`), the serialized FAISS index (`faiss_index/`), pinned `requirements.txt`, `Dockerfile`. |
| `frontend/` | Streamlit UI for staff (`app.py`), `requirements.txt`, `Dockerfile`. |

## Architecture

```
PDF policies ──pymupdf4llm──▶ markdown ──section-aware chunker──▶ 36 chunks
      ──BAAI/bge-small-en-v1.5──▶ FAISS index (serialized to backend/faiss_index)

Staff question ──▶ Streamlit (8501) ──HTTP──▶ Flask (7860)
                         ▲                     │ retrieve top-k chunks
                         │                     │ grounded prompt → LLM
                         └──── answer + cited chunks ◀──┘
```

LLM provider is selected at runtime: an OpenAI-compatible endpoint when
`OPENAI_API_KEY` (and optionally `OPENAI_API_BASE`) is set, otherwise a local
open-weights model (`Qwen3-4B-Instruct-2507`, GGUF via llama.cpp) so the whole
system also runs with no API key at all.

## Run locally with Docker

```bash
cd aurorastay-rag/backend
docker build -t aurora-backend .
docker run --name backend --network host \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" -e OPENAI_API_BASE="$OPENAI_API_BASE" aurora-backend
```

```bash
cd aurorastay-rag/frontend
docker build -t aurora-frontend .
docker run --name frontend --network host aurora-frontend
```

Open http://localhost:8501. Without an API key, omit the `-e` flags and the backend
downloads the local model on first start (about 2.5 GB).

## Deploy with GitHub Codespaces

1. Add `OPENAI_API_KEY` and `OPENAI_API_BASE` as Codespaces secrets with access to this repository.
2. Create a Codespace on this branch, then run the four Docker commands above from
   `aurorastay-rag/backend` and `aurorastay-rag/frontend`.
3. In the **Ports** tab set ports 7860 and 8501 to **Public**. The forwarded 7860 URL is the
   backend URL used in the notebook's *Deployment – Inference* section; the 8501 URL is the staff UI.
4. Delete the Codespace when done to stop consuming quota.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | – | status, models, chunk count |
| `POST /v1/relevant_chunks` | `{query, k}` | top-k chunks with scores and sources |
| `POST /v1/answer` | `{query, k, model_name, temperature, top_p, max_tokens}` | grounded answer |
| `POST /v1/answer_with_relevant_chunks` | same | answer + chunks |
