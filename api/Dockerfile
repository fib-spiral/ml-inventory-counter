# Stage 1: Builder for Python dependencies and system build tools
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install system dependencies required for BUILDING Python packages.
# These will NOT be in the FINAL image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
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
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the installed Python packages from the 'builder' stage.
COPY --from=builder /usr/local /usr/local

# Copy your application code and the model
COPY prediction_api.py .
COPY lambda_handler.py .
COPY models/vegetable_counter_yolov8n_v1.pt ./models/vegetable_counter_yolov8n_v1.pt
RUN mkdir -p models 

# Expose the port
EXPOSE 8000

# Build argument to determine runtime mode (default to fastapi for local builds)
ARG RUNTIME_MODE=fastapi
ENV RUNTIME_MODE=${RUNTIME_MODE}

# Conditional CMD based on build argument
# - If RUNTIME_MODE is "lambda", it runs the RIE with your handler.
# - Otherwise (e.g., "fastapi"), it runs Uvicorn directly.
# More complex but "proper" signal handling
CMD ["sh", "-c", "if [ \"$RUNTIME_MODE\" = \"lambda\" ]; then python -m awslambdaric lambda_handler.handler; else uvicorn prediction_api:app --host 0.0.0.0 --port 8000; fi"]