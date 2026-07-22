# Use official lightweight Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync" \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    RAG_DEMO_MODE=true

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling python packages and running pg tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker layer caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories and files
COPY static/ ./static/
COPY templates/ ./templates/
COPY utils/ ./utils/
COPY retrieval/ ./retrieval/
COPY mini_func/ ./mini_func/
COPY config.py .
COPY connect_db_vllm.py .
COPY mybot_prod_vllm_client.py .
COPY rag_prod_vllm.py .
COPY text_generate_prod_vllm_client.py .

# Create directories for documents and logs
RUN mkdir -p data demo_logs

# Expose server port
EXPOSE 8010

# Command to run application
CMD ["python", "mybot_prod_vllm_client.py"]
