from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Any
import os

# Read the STAGE environment variable set by API Gateway. Default to "" if not found.
# When deployed with a stage, API Gateway often sets 'stage' in requestContext.
# Alternatively, you can set a custom env var.
# For now, let's assume a custom env var for clarity.
stage = os.environ.get('STAGE', "")

# Initialize FastAPI with the root_path set to your stage

app = FastAPI(title="Vegetable Detection API", version="1.0.0", root_path=f"{stage}")

# Class names for the model
CLASS_NAMES = ['carrot', 'bean', 'radish']

# Global model variable
model = None

def load_model():
    """Load the YOLOv8 model"""
    global model
    model_path = "models/vegetable_counter_yolov8n_v1.pt"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = YOLO(model_path)
    return model

def validate_image(file: UploadFile) -> bool:
    """Validate if the uploaded file is an image"""
    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ""
    
    if file_extension not in allowed_extensions:
        return False
    
    # Check MIME type
    if not file.content_type or not file.content_type.startswith('image/'):
        return False
    
    return True

def process_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded file bytes to OpenCV image format"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert PIL to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def format_predictions(results) -> List[Dict[str, Any]]:
    """Format YOLO predictions into API response format"""
    detections = []
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # class IDs
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                class_name = CLASS_NAMES[int(cls_id)]
                
                prediction = {
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bounding_box": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                }
                detections.append(prediction)
    
    return detections

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict vegetables in uploaded image
    
    Returns:
        JSON response with detected vegetables, their confidence scores, and bounding boxes
    """
    try:
        # Validate file is an image
        if not validate_image(file):
            raise HTTPException(
                status_code=400, 
                detail="Uploaded file is not a valid image. Please upload a JPG, PNG, or similar image file."
            )
        
        # Read file contents
        try:
            file_bytes = await file.read()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Failed to read uploaded file."
            )
        
        # Process image
        try:
            image = process_image(file_bytes)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image format."
            )
        
        # Run inference
        try:
            results = model(image)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Model inference failed. Please try again."
            )
        
        # Format predictions
        predictions = format_predictions(results)
        
        # Count vegetables by class
        vegetable_counts = {class_name: 0 for class_name in CLASS_NAMES}
        for pred in predictions:
            vegetable_counts[pred["class_name"]] += 1
        
        response = {
            "predictions": predictions,
            "total_detections": len(predictions),
            "vegetable_counts": vegetable_counts
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction."
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)