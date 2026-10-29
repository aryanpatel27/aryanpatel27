# ---------------------------------------------------------------------------
# AuroraStay policy assistant - Streamlit frontend image
# Build:  docker build -t aurora-frontend .
# Run:    docker run --name frontend --network host aurora-frontend
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Install the two runtime dependencies first so the layer is cached.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Backend location; override with -e BACKEND_URL=... when not on the host network.
ENV BACKEND_URL=http://localhost:7860 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# XSRF protection is disabled because Codespaces port-forwarding rewrites the
# origin header, which would otherwise block form submissions.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.headless=true"]
