FROM python:3.11-slim

LABEL maintainer="Interview Trainer Agent"
LABEL description="RAG-powered Interview Preparation Assistant using IBM Granite on watsonx.ai"

WORKDIR /app

# System dependencies for chromadb / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY run.py .

# Create data directory for ChromaDB persistence
RUN mkdir -p ./data/chroma_db

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run
CMD ["python", "run.py"]
