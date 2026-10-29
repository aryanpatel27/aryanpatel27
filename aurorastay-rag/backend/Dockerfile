# ---------------------------------------------------------------------------
# AuroraStay policy assistant - Flask backend image
#
# Build:  docker build -t aurora-backend .
# Run (OpenAI-compatible LLM, recommended in Codespaces):
#   docker run --name backend --network host \
#       -e OPENAI_API_KEY="$OPENAI_API_KEY" -e OPENAI_API_BASE="$OPENAI_API_BASE" aurora-backend
# Run (no API key -> local open-weights LLM; downloads a ~2.5 GB GGUF on first use):
#   docker run --name backend --network host aurora-backend
# ---------------------------------------------------------------------------
# Step 1 - small base image
FROM python:3.12-slim

WORKDIR /app

# Build tools are only needed to compile llama-cpp-python (local LLM fallback).
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

# Step 2 - CPU-only PyTorch (a fraction of the size of the default CUDA build)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# Step 3 - remaining pinned dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4 - application code + serialized FAISS index
COPY rag_utils.py app.py ./
COPY faiss_index ./faiss_index

# Step 5 - bake the embedding model into the image so start-up needs no download
ENV EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# Optional: bake the local GGUF LLM too (docker build --build-arg BAKE_LOCAL_LLM=true ...)
ARG BAKE_LOCAL_LLM=false
ENV LOCAL_LLM_REPO=unsloth/Qwen3-4B-Instruct-2507-GGUF \
    LOCAL_LLM_FILE=Qwen3-4B-Instruct-2507-Q4_K_M.gguf
RUN if [ "$BAKE_LOCAL_LLM" = "true" ]; then \
      python -c "import os; from huggingface_hub import hf_hub_download; hf_hub_download(os.environ['LOCAL_LLM_REPO'], os.environ['LOCAL_LLM_FILE'])"; \
    fi

EXPOSE 7860

# Step 6 - production WSGI server. One worker (models are memory-heavy); long
# timeout because local-LLM generation on a 2-core Codespace can take minutes.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "600", "app:app"]
