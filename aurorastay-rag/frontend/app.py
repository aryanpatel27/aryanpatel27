"""
app.py - Streamlit user interface for AuroraStay hotel staff.

The UI never touches models or documents directly: it sends the question and
the chosen parameters to the Flask backend and renders the JSON response.
Backend location comes from the BACKEND_URL environment variable (default:
http://localhost:7860, which works when both containers share the host network).
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("BACKEND_TIMEOUT", "600"))  # matches the backend gunicorn timeout

st.set_page_config(page_title="AuroraStay Policy Assistant", page_icon="🏨", layout="wide")

# ---------------------------------------------------------------- sidebar --- #
with st.sidebar:
    st.header("⚙️ Parameters")
    k = st.slider("k (chunks to retrieve)", 1, 10, 4,
                  help="How many policy excerpts are passed to the language model.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05,
                            help="0 = deterministic, policy-faithful answers (recommended).")
    top_p = st.slider("Top-p", 0.1, 1.0, 1.0, 0.05, help="Nucleus sampling cut-off.")
    max_tokens = st.slider("Max tokens", 64, 1024, 512, 32, help="Upper bound on answer length.")
    model_name = st.text_input("Model name (API provider only)", value="gpt-4o-mini",
                               help="Ignored when the backend runs the local open-weights model.")
    st.divider()
    st.caption(f"Backend: `{BACKEND_URL}`")
    if st.button("Check backend"):
        try:
            h = requests.get(f"{BACKEND_URL}/health", timeout=15).json()
            st.success(f"OK · {h.get('chunks_indexed')} chunks · LLM: {h.get('llm')}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backend not reachable: {exc}")

# ------------------------------------------------------------------- main --- #
st.title("🏨 AuroraStay Policy Assistant")
st.write("Ask a question about AuroraStay booking, cancellation, refund, room, "
         "amenity, pet, parking or guest-conduct policies. Answers are grounded only "
         "in the official policy documents and show their sources.")

mode = st.radio("Response mode", ["Answer only", "Answer + Relevant Chunks"], horizontal=True)

with st.expander("Example questions"):
    st.markdown(
        "- What is the pet charge at ASH-CHI Loop?\n"
        "- Is breakfast free at the Miami Brickell property?\n"
        "- I have a flexible reservation. How late can I cancel without being charged?\n"
        "- A guest has a service dog at ASH-BOS Harbor. Do they pay the pet fee and what can we ask?\n"
        "- The pending hold on a guest's card is larger than the final bill. When is the difference released?"
    )

query = st.text_input("Your question", placeholder="e.g., Is breakfast free at the Miami Brickell property?")

if st.button("Submit", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        endpoint = "/v1/answer" if mode == "Answer only" else "/v1/answer_with_relevant_chunks"
        payload = {"query": query.strip(), "k": k, "model_name": model_name,
                   "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        with st.spinner("Retrieving policy excerpts and generating the answer..."):
            try:
                resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=REQUEST_TIMEOUT)
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is the backend container running?")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("The backend took too long to respond. Try again or lower max tokens.")
                st.stop()

        if resp.status_code != 200:
            try:
                st.error(f"Backend error {resp.status_code}: {resp.json().get('error', resp.text)}")
            except ValueError:
                st.error(f"Backend error {resp.status_code}: {resp.text[:500]}")
            st.stop()

        data = resp.json()
        st.subheader("Answer")
        # Escape "$" so dollar amounts are not interpreted as LaTeX math by Streamlit's markdown renderer.
        st.markdown(data.get("answer", "").replace("$", r"\$"))

        timing = data.get("timing_seconds", {})
        params = data.get("parameters", {})
        st.caption(f"Model: {params.get('model')} · k={params.get('k')} · "
                   f"retrieval {timing.get('retrieval', '?')}s · generation {timing.get('generation', '?')}s")

        if mode == "Answer + Relevant Chunks":
            st.subheader("Relevant Chunks")
            chunks = data.get("relevant_chunks", [])
            if not chunks:
                st.info("No chunks were returned.")
            for c in chunks:
                label = f"Chunk {c.get('rank')} · {c.get('doc_title')} ({c.get('policy_code')}) · {c.get('section')} · score {c.get('score')}"
                with st.expander(label):
                    st.markdown(f"**Source file:** `{c.get('source')}`")
                    st.markdown("**Content:**")
                    st.markdown(c.get("content", "").replace("$", r"\$"))
