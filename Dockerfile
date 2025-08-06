# Multi-stage Dockerfile for AI-RAG-Project with GPU Support and Model Pre-loading
# =================================================================================

# ============================
# Stage 1: CPU Model Builder
# ============================
FROM python:3.9-slim as cpu-model-builder

# Install system dependencies for model building
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies (auto-detects CPU/GPU)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directories
RUN mkdir -p /app/models/transformers /app/models/huggingface /app/models/torch

# Set model cache environment variables
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch
ENV MODEL_DEVICE=cpu

# Copy source code for model warm-up
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Pre-load models with CPU optimization
RUN echo "🔥 Pre-loading AI models (CPU mode) for instant startup..." && \
    python scripts/warm_up_models.py && \
    echo "✅ CPU model pre-loading complete!"

# ============================
# Stage 2: GPU Model Builder  
# ============================
FROM nvidia/cuda:12.1-devel-ubuntu20.04 as gpu-model-builder

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    git \
    wget \
    curl \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies (auto-detects GPU acceleration)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directories
RUN mkdir -p /app/models/transformers /app/models/huggingface /app/models/torch

# Set model cache environment variables
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch
ENV MODEL_DEVICE=cuda
ENV CUDA_VISIBLE_DEVICES=0

# Copy source code for model warm-up
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Pre-load models with GPU acceleration
RUN echo "🚀 Pre-loading AI models (GPU mode) for instant startup..." && \
    python scripts/warm_up_models.py && \
    echo "✅ GPU model pre-loading complete!"

# ===============================
# Stage 3: CPU Production Runtime
# ===============================
FROM python:3.9-slim as cpu-production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-loaded models from CPU builder stage
COPY --from=cpu-model-builder --chown=appuser:appuser /app/models /app/models

# Copy application code
COPY --chown=appuser:appuser . .

# Set model cache environment variables to use pre-loaded models
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch
ENV MODEL_DEVICE=cpu

# Set application environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Docker optimization environment variables
ENV DOCKER_PRELOADED_MODELS=true
ENV RAG_DATA_DIR=/app/data
ENV RAG_CACHE_DIR=/app/cache

# GPU memory optimization environment variables for RTX 1060 6GB
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,roundup_power2_divisions:8

# Create data and cache directories with proper ownership
RUN mkdir -p /app/data /app/cache /app/logs && \
    chown -R appuser:appuser /app/data /app/cache /app/logs

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

LABEL maintainer="AI-RAG-Project" \
      version="2.0-cpu" \
      description="CPU-optimized AI RAG application with ColPali visual understanding"

# Enhanced health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Default command with optimized settings
CMD ["streamlit", "run", "streamlit_rag_app.py", \
     "--server.headless", "true", \
     "--server.fileWatcherType", "none", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]

# ===============================
# Stage 4: GPU Production Runtime
# ===============================
FROM nvidia/cuda:12.1-runtime-ubuntu20.04 as gpu-production

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy and install consolidated requirements (GPU auto-detected)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-loaded models from GPU builder stage
COPY --from=gpu-model-builder --chown=appuser:appuser /app/models /app/models

# Copy application code
COPY --chown=appuser:appuser . .

# Set model cache environment variables for GPU
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch
ENV MODEL_DEVICE=cuda
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set application environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Docker optimization environment variables
ENV DOCKER_PRELOADED_MODELS=true
ENV RAG_DATA_DIR=/app/data
ENV RAG_CACHE_DIR=/app/cache

# GPU memory optimization environment variables for RTX 1060 6GB
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,roundup_power2_divisions:8

# Create data and cache directories with proper ownership
RUN mkdir -p /app/data /app/cache /app/logs && \
    chown -R appuser:appuser /app/data /app/cache /app/logs

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

LABEL maintainer="AI-RAG-Project" \
      version="2.0-gpu" \
      description="GPU-accelerated AI RAG application with ColPali visual understanding"

# Enhanced health check with GPU validation
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Default command with GPU-optimized settings
CMD ["streamlit", "run", "streamlit_rag_app.py", \
     "--server.headless", "true", \
     "--server.fileWatcherType", "none", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]

# =================================
# Build and Deployment Instructions
# =================================
# 
# CPU Build:
#   docker build --target cpu-production -t ai-rag-app:cpu .
# 
# GPU Build:
#   docker build --target gpu-production -t ai-rag-app:gpu .
# 
# Development (model builder only):
#   docker build --target cpu-model-builder -t ai-rag-models:cpu .
#   docker build --target gpu-model-builder -t ai-rag-models:gpu .
# 
# Quick Start:
#   CPU: docker run -p 8501:8501 ai-rag-app:cpu
#   GPU: docker run --gpus all -p 8501:8501 ai-rag-app:gpu
# 
# Production Deployment:
#   Use docker-compose.yml for orchestrated deployment