# Multi-stage Dockerfile for AI-RAG-Project with Model Pre-loading
# Stage 1: Model Download and Caching
FROM python:3.9-slim as model-builder

# Install system dependencies for model building
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
# Note: poppler-utils provides cross-platform PDF processing support

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directories
RUN mkdir -p /app/models/transformers /app/models/huggingface /app/models/torch

# Set model cache environment variables
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch

# Copy source code for model warm-up
COPY src/ ./src/
COPY scripts/ ./scripts/

# Make warm-up script executable
RUN chmod +x scripts/warm_up_models.py

# Pre-load all models during build time
RUN echo "🔥 Pre-loading AI models for instant startup..." && \
    python scripts/warm_up_models.py && \
    echo "✅ Model pre-loading complete!"

# Stage 2: Production Runtime
FROM python:3.9-slim as production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
# Note: poppler-utils enables ColPali visual document processing with graceful fallback

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-loaded models from builder stage
COPY --from=model-builder /app/models /app/models

# Copy application code
COPY . .

# Set model cache environment variables to use pre-loaded models
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch

# Set application environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Docker optimization environment variables
ENV DOCKER_PRELOADED_MODELS=true
ENV RAG_DATA_DIR=/app/data
ENV RAG_CACHE_DIR=/app/cache

# Create data and cache directories
RUN mkdir -p /app/data /app/cache

# Expose Streamlit port
EXPOSE 8501

# Health check to verify models are available
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/models/model_manifest.json') else 1)"

# Default command
CMD ["streamlit", "run", "streamlit_rag_app.py", "--server.headless", "true", "--server.fileWatcherType", "none"]

# GPU Support Variant (uncomment for GPU builds)
# FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-base
# ... [GPU-specific configuration would go here]

# Build instructions:
# For CPU: docker build -t ai-rag-app .
# For GPU: docker build -t ai-rag-app:gpu --target gpu-production .
# Quick start: docker run -p 8501:8501 ai-rag-app