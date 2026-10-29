"""
rag_utils.py - Core Retrieval-Augmented Generation pipeline for the AuroraStay
Hotels policy assistant.

The same functions are used inside the development notebook and inside the
deployed Flask backend, so behaviour is identical in both places:

    PDF  ->  markdown (pymupdf4llm)  ->  section-aware chunks  ->  embeddings
         ->  FAISS index  ->  retrieve(k)  ->  grounded prompt  ->  LLM answer

LLM provider selection (in order):
  1. OPENAI_API_KEY is set   -> OpenAI-compatible chat API (OPENAI_API_BASE optional)
  2. otherwise               -> local open-weights model served by llama.cpp
                                (LOCAL_LLM_REPO / LOCAL_LLM_FILE, default Qwen3-4B-Instruct-2507)
Credentials are read from environment variables only - never from source files.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Configuration (overridable through environment variables)
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_LLM_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")
LOCAL_LLM_REPO = os.getenv("LOCAL_LLM_REPO", "unsloth/Qwen3-4B-Instruct-2507-GGUF")
LOCAL_LLM_FILE = os.getenv("LOCAL_LLM_FILE", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH")  # explicit path to a .gguf file
LOCAL_LLM_CTX = int(os.getenv("LOCAL_LLM_CTX", "8192"))

CHUNK_MAX_CHARS = 1400   # upper bound of a chunk (characters)
CHUNK_OVERLAP = 150      # overlap between consecutive sub-chunks of one section

# Footer / page-number lines that pymupdf4llm carries over from every page.
_FOOTER_RE = re.compile(r"^AuroraStay Hotels \| AS-\d+ \| Internal operating policy\s*$", re.M)
_PAGE_RE = re.compile(r"^Page \d+\s*$", re.M)
_TITLE_RE = re.compile(r"^# \*\*(.+?)\*\*\s*$", re.M)
_CODE_RE = re.compile(r"\|\*\*Policy code\*\*\|(AS-\d+)\|")
_H2_RE = re.compile(r"^## \*\*(.+?)\*\*\s*$", re.M)
# Boilerplate repeated verbatim in all four documents (adds no answerable facts).
_DOCMAP_RE = re.compile(r"^### \*\*Document map\*\*.*?(?=^#|\Z)", re.M | re.S)
_OPNOTE_RE = re.compile(r"^Operating note:.*?$", re.M)
_BANNER_RE = re.compile(r"^AURORASTAY HOTELS\s*$", re.M)


# --------------------------------------------------------------------------- #
# 1. Document loading
# --------------------------------------------------------------------------- #
def pdf_to_markdown(pdf_path: str) -> str:
    """Convert a PDF to GitHub-flavoured markdown (headings + pipe tables)."""
    import pymupdf4llm  # imported lazily: not needed inside the backend container

    return pymupdf4llm.to_markdown(pdf_path, show_progress=False)


def clean_markdown(md: str) -> str:
    """Remove running footers / page numbers and collapse blank lines."""
    md = _FOOTER_RE.sub("", md)
    md = _PAGE_RE.sub("", md)
    md = _DOCMAP_RE.sub("", md)
    md = _OPNOTE_RE.sub("", md)
    md = _BANNER_RE.sub("", md)
    md = md.replace("<br>", " ")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def document_info(md: str) -> Dict[str, str]:
    """Extract the document title and policy code (e.g. AS-02) from markdown."""
    title = _TITLE_RE.search(md)
    code = _CODE_RE.search(md)
    return {
        "title": title.group(1).strip() if title else "Unknown document",
        "policy_code": code.group(1) if code else "",
    }


def load_documents(data_dir: str) -> List[Dict[str, Any]]:
    """Load every policy PDF in `data_dir` and return parsed markdown + metadata."""
    docs = []
    for pdf in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        md = clean_markdown(pdf_to_markdown(pdf))
        info = document_info(md)
        docs.append({"source": os.path.basename(pdf), "markdown": md, **info})
    return docs


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #
def _split_sections(md: str) -> List[Dict[str, str]]:
    """Split markdown on level-2 headings. Text before the first heading becomes 'Overview'."""
    positions = [(m.start(), m.end(), m.group(1).strip()) for m in _H2_RE.finditer(md)]
    sections = []
    if not positions or positions[0][0] > 0:
        head = md[: positions[0][0]] if positions else md
        sections.append({"section": "Overview", "text": head.strip()})
    for i, (start, end, name) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(md)
        sections.append({"section": name, "text": md[end:stop].strip()})
    return [s for s in sections if s["text"]]


def _split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Split a long section into sub-chunks on paragraph boundaries.

    Markdown tables never contain blank lines, so a table always stays inside
    one sub-chunk unless the table itself is longer than `max_chars`.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    return splitter.split_text(text)


def chunk_document(doc: Dict[str, Any], max_chars: int = CHUNK_MAX_CHARS,
                   overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Section-aware chunker.

    * One section (## heading) -> one chunk when it fits in `max_chars`.
    * Longer sections are split on paragraph boundaries with overlap.
    * Every chunk is prefixed with a breadcrumb (document + section) so the
      embedding carries the context of where the text came from and the LLM
      can cite it.
    """
    chunks = []
    for sec in _split_sections(doc["markdown"]):
        pieces = [sec["text"]] if len(sec["text"]) <= max_chars else _split_long_text(sec["text"], max_chars, overlap)
        for part_no, piece in enumerate(pieces, start=1):
            breadcrumb = f"Document: {doc['title']} ({doc['policy_code']}) | Section: {sec['section']}"
            if len(pieces) > 1:
                breadcrumb += f" (part {part_no}/{len(pieces)})"
            chunks.append({
                "text": f"{breadcrumb}\n\n{piece}",
                "metadata": {
                    "source": doc["source"],
                    "doc_title": doc["title"],
                    "policy_code": doc["policy_code"],
                    "section": sec["section"],
                    "part": part_no,
                    "n_parts": len(pieces),
                },
            })
    return chunks


def chunk_documents(docs: List[Dict[str, Any]], **kw) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        out.extend(chunk_document(d, **kw))
    for i, c in enumerate(out):
        c["metadata"]["chunk_id"] = i
    return out


# --------------------------------------------------------------------------- #
# 3. Embeddings + vector store (FAISS)
# --------------------------------------------------------------------------- #
def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME):
    """Sentence-transformers embedding model wrapped for LangChain.

    Embeddings are L2-normalised so FAISS inner-product == cosine similarity.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def build_vectorstore(chunks: List[Dict[str, Any]], embeddings):
    """Embed all chunks and index them in FAISS (exact inner-product search)."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.documents import Document

    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
    return FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


def save_vectorstore(vectorstore, path: str) -> None:
    vectorstore.save_local(path)


def load_vectorstore(path: str, embeddings):
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# --------------------------------------------------------------------------- #
# 4. Retrieval  (hybrid: dense cosine similarity + BM25 keyword search, fused with RRF)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\$\.:]*")
_BM25_CACHE: Dict[int, Any] = {}
RRF_K = 60            # standard reciprocal-rank-fusion constant
CANDIDATES_PER_RETRIEVER = 10


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_for(vectorstore):
    """Build (once per vectorstore) a BM25 index over the chunks stored in FAISS."""
    key = id(vectorstore)
    if key not in _BM25_CACHE:
        from rank_bm25 import BM25Okapi

        docs = list(vectorstore.docstore._dict.values())
        _BM25_CACHE[key] = (docs, BM25Okapi([_tokenize(d.page_content) for d in docs]))
    return _BM25_CACHE[key]


def _to_result(doc, rank: int, score: float, dense_score: Optional[float]) -> Dict[str, Any]:
    return {
        "rank": rank,
        "score": round(float(score), 4),
        "dense_similarity": None if dense_score is None else round(float(dense_score), 4),
        "content": doc.page_content,
        "source": doc.metadata.get("source"),
        "doc_title": doc.metadata.get("doc_title"),
        "policy_code": doc.metadata.get("policy_code"),
        "section": doc.metadata.get("section"),
        "chunk_id": doc.metadata.get("chunk_id"),
    }


def retrieve_dense(query: str, k: int, vectorstore) -> List[Dict[str, Any]]:
    """Pure embedding search (cosine similarity); kept for comparison / ablation."""
    hits = vectorstore.similarity_search_with_score(query, k=k)
    return [_to_result(doc, r, score, score) for r, (doc, score) in enumerate(hits, start=1)]


def retrieve(query: str, k: int, vectorstore, mode: str = "hybrid") -> List[Dict[str, Any]]:
    """Return the top-k chunks for `query`.

    mode="hybrid" (default): top-10 dense + top-10 BM25 candidates are merged with
    reciprocal-rank fusion  score = sum(1 / (60 + rank_i)).  Dense search captures
    paraphrases ("loud since midnight" -> quiet hours); BM25 captures exact tokens
    such as property codes (ASH-CHI) and dollar amounts.
    mode="dense": embedding search only.
    """
    if mode == "dense":
        return retrieve_dense(query, k, vectorstore)

    n = max(CANDIDATES_PER_RETRIEVER, k)
    dense_hits = vectorstore.similarity_search_with_score(query, k=n)
    docs, bm25 = _bm25_for(vectorstore)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_order = sorted(range(len(docs)), key=lambda i: -bm25_scores[i])[:n]

    fused: Dict[int, float] = {}
    by_id: Dict[int, Any] = {}
    dense_sim: Dict[int, float] = {}
    for rank, (doc, sim) in enumerate(dense_hits, start=1):
        cid = doc.metadata["chunk_id"]
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        by_id[cid] = doc
        dense_sim[cid] = float(sim)
    for rank, i in enumerate(bm25_order, start=1):
        if bm25_scores[i] <= 0:
            continue
        cid = docs[i].metadata["chunk_id"]
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        by_id.setdefault(cid, docs[i])

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [_to_result(by_id[cid], r, fused[cid], dense_sim.get(cid)) for r, cid in enumerate(top, start=1)]


# --------------------------------------------------------------------------- #
# 5. Generation
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are the AuroraStay Hotels policy assistant. You help front-desk and guest-services staff answer guest questions using ONLY the official AuroraStay policy excerpts provided to you.

Answer rules:
1. Use only the information in the provided policy excerpts. Do not use outside knowledge or assumptions.
2. If the excerpts do not contain the information needed, say clearly that the provided policies do not cover it and tell the staff member to check the live property profile or escalate to the manager on duty. Never guess a fee, time or rule.
3. Quote exact figures, times and deadlines as written (e.g. "$50 per night, $150 stay cap", "6:00 PM local property time").
4. Property-specific values (ASH-ATL Midtown, ASH-BOS Harbor, ASH-CHI Loop, ASH-DEN Union, ASH-MIA Brickell, ASH-SEA Pike) apply only to that property. Never transfer a fee or time from one property to another.
5. If different excerpts conflict, mention the conflict and say that the reservation confirmation or live property profile controls.
6. Mention relevant exceptions, verification steps or escalation paths that the policy requires before staff make a promise to a guest.
7. Keep the answer concise: 2-5 sentences, written so the staff member can relay it to the guest.
8. Do not mention "excerpts", "chunks" or "context" to the user; refer to the policies by their document name when helpful (e.g. "the Booking Terms and Cancellation Policy").
"""


def build_messages(query: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Assemble the chat messages: system rules + numbered policy excerpts + question."""
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(f"[Excerpt {c['rank']}]\n{c['content']}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no excerpts retrieved)"
    user = (
        "Policy excerpts:\n\n"
        f"{context}\n\n"
        f"Staff question: {query}\n\n"
        "Answer the staff question using only the excerpts above."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ---- LLM providers --------------------------------------------------------- #
_LOCAL_LLM = None  # cached llama.cpp instance (loading takes a few seconds)


def llm_provider() -> str:
    """'openai' when an API key is configured, otherwise 'local'."""
    return "openai" if os.getenv("OPENAI_API_KEY") else "local"


def _resolve_local_model_path() -> str:
    if LOCAL_LLM_PATH and os.path.exists(LOCAL_LLM_PATH):
        return LOCAL_LLM_PATH
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=LOCAL_LLM_REPO, filename=LOCAL_LLM_FILE)


def get_local_llm():
    """Load (once) the local GGUF model with llama.cpp."""
    global _LOCAL_LLM
    if _LOCAL_LLM is None:
        from llama_cpp import Llama

        _LOCAL_LLM = Llama(
            model_path=_resolve_local_model_path(),
            n_ctx=LOCAL_LLM_CTX,
            n_threads=os.cpu_count() or 2,
            n_batch=512,
            verbose=False,
        )
    return _LOCAL_LLM


def chat_completion(messages: List[Dict[str, str]], model_name: str = DEFAULT_LLM_NAME,
                    temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 512,
                    response_format: Optional[Dict[str, Any]] = None) -> str:
    """Provider-agnostic chat completion returning the assistant text."""
    if llm_provider() == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE") or None)
        kwargs = dict(model=model_name, messages=messages, temperature=temperature,
                      top_p=top_p, max_tokens=max_tokens)
        if response_format:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    llm = get_local_llm()
    kwargs = dict(messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    if response_format:
        kwargs["response_format"] = response_format
    resp = llm.create_chat_completion(**kwargs)
    return resp["choices"][0]["message"]["content"].strip()


def active_llm_name(model_name: str = DEFAULT_LLM_NAME) -> str:
    return model_name if llm_provider() == "openai" else f"local:{LOCAL_LLM_FILE}"


def generate(query: str, retrieved_chunks: List[Dict[str, Any]], model_name: str = DEFAULT_LLM_NAME,
             temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 512) -> str:
    """Generate a grounded answer from the retrieved chunks."""
    messages = build_messages(query, retrieved_chunks)
    return chat_completion(messages, model_name=model_name, temperature=temperature,
                           top_p=top_p, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# 6. End-to-end RAG
# --------------------------------------------------------------------------- #
def rag(query: str, k: int = 4, model_name: str = DEFAULT_LLM_NAME, temperature: float = 0.0,
        top_p: float = 1.0, max_tokens: int = 512, vectorstore=None) -> Dict[str, Any]:
    """Retrieve top-k chunks, generate an answer, and return both with timings."""
    t0 = time.time()
    chunks = retrieve(query, k, vectorstore)
    t1 = time.time()
    answer = generate(query, chunks, model_name=model_name, temperature=temperature,
                      top_p=top_p, max_tokens=max_tokens)
    t2 = time.time()
    return {
        "query": query,
        "answer": answer,
        "relevant_chunks": chunks,
        "parameters": {"k": k, "model": active_llm_name(model_name), "temperature": temperature,
                       "top_p": top_p, "max_tokens": max_tokens},
        "timing_seconds": {"retrieval": round(t1 - t0, 3), "generation": round(t2 - t1, 3)},
    }


__all__ = [
    "load_documents", "pdf_to_markdown", "clean_markdown", "document_info",
    "chunk_document", "chunk_documents", "get_embeddings", "build_vectorstore",
    "save_vectorstore", "load_vectorstore", "retrieve", "retrieve_dense", "build_messages",
    "chat_completion", "generate", "rag", "llm_provider", "active_llm_name",
    "EMBEDDING_MODEL_NAME", "DEFAULT_LLM_NAME", "SYSTEM_PROMPT",
]
