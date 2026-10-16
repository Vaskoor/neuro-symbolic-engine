# =============================================================================
# Dockerfile — Neuro-Symbolic Reasoning Engine
# Lightweight, offline-runnable API image.
# =============================================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    NEUROSYM_ONTOLOGY=/app/config/ontology.ttl \
    NEUROSYM_RULES=/app/config/rules.yaml \
    NEUROSYM_HOST=0.0.0.0 \
    NEUROSYM_PORT=8000 \
    NEUROSYM_LLM_ENABLED=0

WORKDIR /app

# --- 1. Install dependencies first (better layer caching) -------------------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- 2. Copy source + config -------------------------------------------------
COPY src/    ./src/
COPY config/ ./config/

# Install the package itself so `import neurosym.*` resolves cleanly
RUN pip install --no-cache-dir --no-deps -e .

# --- 3. Runtime ---------------------------------------------------------------
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "neurosym.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
