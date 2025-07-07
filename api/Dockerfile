# Stage 1: Builder for Python dependencies and system build tools
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install system dependencies required for BUILDING Python packages.
# These will NOT be in the FINAL image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    # Other build-time deps you might need that aren't for runtime:
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt for pip install
COPY requirements-api.txt .

# Install Python dependencies.
# Install them into /usr/local, which is where Python typically installs site-packages.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt \
    --prefix /usr/local \
    --extra-index-url https://download.pytorch.org/whl/cpu

# --- Stage 2: Final Runtime Image ---
# Start from a fresh, minimal Python slim image again.
FROM python:3.11-slim-bookworm

# Set working directory for the final application
WORKDIR /app

# **CRITICAL FIX**: Install RUNTIME system dependencies here.
# These are the shared libraries that Python packages (like OpenCV) need to RUN.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages from the 'builder' stage.
COPY --from=builder /usr/local /usr/local

# Copy your application code and the model
COPY api.py .
RUN mkdir -p models 
COPY models/vegetable_counter_yolov8n_v1.pt ./models/vegetable_counter_yolov8n_v1.pt

# Expose the port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]