# src/api.py
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
from ultralytics import YOLO
import numpy as np # For potential image processing

app = FastAPI(title="Vegetable Counter API")

# Load your trained model
# Ensure this path is correct relative to where the Docker container will run the app
# If running from project root, it's 'models/vegetable_counter_yolov8n_v1.pt'
MODEL_PATH = "models/vegetable_counter_yolov8n_v1.pt" # Adjust if your model path is different

try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    # In a real app, you might want to handle this more gracefully or fail startup.
    model = None # Set model to None if loading fails

@app.get("/")
async def root():
    return {"message": "Welcome to the Vegetable Counter API! Use /predict to upload an image."}

@app.post("/predict")
async def predict_items(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    # 1. Read image from upload
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # 2. Perform inference with YOLOv8
    # The 'predict' method returns a list of Results objects
    # verbose=False to suppress print output to console
    results = model.predict(source=image, conf=0.25, iou=0.7, verbose=False) # conf and iou are thresholds

    detections = []
    counts = defaultdict(int)

    # 3. Process results
    for r in results:
        # Boxes: xyxy (coordinates), conf (confidence), cls (class ID)
        # You'll need your CLASS_NAMES list here to map class_id to name
        # For simplicity, we'll assume it's available or hardcoded for the API
        # Ideally, pass class_names to the API or load from data.yaml
        
        # This example needs your CLASS_NAMES as defined in prepare_dataset.py
        # For this simple API, let's hardcode it for demonstration.
        CLASS_NAMES_API = ['carrot', 'bean', 'radish'] # Make sure this matches your data.yaml

        for box in r.boxes:
            class_id = int(box.cls.item())
            confidence = round(box.conf.item(), 2)
            
            # Convert tensor to list for JSON serialization
            xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            class_name = CLASS_NAMES_API[class_id] if class_id < len(CLASS_NAMES_API) else "unknown"
            
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "box": {
                    "x1": round(xyxy[0]), "y1": round(xyxy[1]),
                    "x2": round(xyxy[2]), "y2": round(xyxy[3])
                }
            })
            counts[class_name] += 1

    return {
        "detections": detections,
        "counts": dict(counts), # Convert defaultdict to dict for JSON
        "total_items": sum(counts.values())
    }