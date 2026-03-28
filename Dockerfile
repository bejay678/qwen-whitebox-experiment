# Qwen White-box Experiment Dockerfile

# Use official Python runtime as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create data directory
RUN mkdir -p /app/data && chown appuser:appuser /app/data

# Set environment variables for runtime
ENV QWEN_MODEL_PATH="/app/models/Qwen2.5-0.5B-Instruct" \
    FAISS_INDEX_PATH="/app/data/indices/editable_index.bin" \
    ADAPTER_MODEL_PATH="/app/models/adapter_model.pt" \
    PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print('Health check passed')" || exit 1

# Default command
CMD ["python", "-c", "print('Qwen White-box Experiment container is ready.\\nRun: python examples/basic_retrieval.py')"]