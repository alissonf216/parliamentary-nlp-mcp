# Parliamentary NLP MCP Auditor — reproducible inference image
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    PARLIAMENTARY_NLP_MODEL_ID=alissonf216/parliamentary-bertimbau-auditor

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip setuptools wheel \
    && pip install .

# Persist Hugging Face downloads across container restarts
RUN mkdir -p /cache/huggingface && chmod -R 777 /cache

VOLUME ["/cache/huggingface"]

# MCP speaks JSON-RPC over stdio — keep stdin attached (docker run -i / compose stdin_open)
ENTRYPOINT ["parliamentary-nlp-mcp"]
