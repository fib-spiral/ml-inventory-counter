# Dockerfile.frontend
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy requirements for the frontend app
# Create a new, minimal requirements-frontend.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code
COPY app.py .
COPY drawing.py .
COPY api_client.py .

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]