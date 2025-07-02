# Use a lightweight Python base image
FROM python:3.11-bookworm 

# Set working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV (cv2)
# The 'install_recommends false' helps keep the image smaller by avoiding suggested packages.
# rm -rf /var/lib/apt/lists/* cleans up apt cache to keep image size down.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libsm6 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy your trained model
# This assumes models/vegetable_counter_yolov8n_v1.pt is in your project root
# Adjust if your models directory is named differently or model file path varies.
COPY models/vegetable_counter_yolov8n_v1.pt ./models/vegetable_counter_yolov8n_v1.pt

# Copy your application code
# This copies the entire 'src' directory into /app/src
COPY src/ ./src/

# Command to run the FastAPI application with Uvicorn
# Host 0.0.0.0 makes it accessible from outside the container
# Port 8000 is the default for Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]