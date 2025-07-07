import os
import mlflow
import subprocess
import json
import torch

# --- Check for GPU Availability ---
if torch.cuda.is_available():
    print("CUDA GPU is available!")
elif torch.backends.mps.is_available():
    print("Apple MPS (GPU) is available!")
else:
    print("No GPU detected, training on CPU.")
from ultralytics import YOLO

# --- Configuration ---
# Path to your data.yaml file
DATA_YAML_PATH = 'data/data.yaml'

# Path to save MLflow artifacts (models, plots) relative to the experiment run
MLFLOW_ARTIFACT_PATH = 'yolov8_artifacts'

# Choose a pre-trained YOLOv8 model variant
YOLOV8_MODEL_VARIANT = 'yolov8n.pt'

# Training parameters
EPOCHS = 10
BATCH_SIZE = 8
IMG_SIZE = 640

# --- Main Training Logic ---
def train_yolov8_model():
    # 1. Get DVC data hash (optional, but good for linking)
    data_dvc_hash = "N/A" # Default if DVC lookup fails
    try:
 # Use dvc status --json to get information about tracked files
        # This is more robust than dvc diff for simply getting the current hash.
        result = subprocess.run(['dvc', 'status', '--json', 'data/'], capture_output=True, text=True, check=True)
        dvc_status_output = json.loads(result.stdout)

                # dvc status --json returns a list of dictionaries for each tracked item.
        # We need to find the one corresponding to 'data/'.
        # The 'hash' field within the 'path' entry is what we're looking for.
        
        # Example structure of dvc_status_output:
        # [
        #   {
        #     "path": "data",
        #     "hash": "dvc_file_hash_here",
        #     "status": "up to date",
        #     "is_dir": True,
        #     "name": "data.dvc"
        #   }
        # ]
        
        for item in dvc_status_output:
            if item.get('path') == 'data' and 'hash' in item:
                data_dvc_hash = item['hash']
                break
        
        if data_dvc_hash != "N/A":
            print(f"Using DVC data hash: {data_dvc_hash}")
        else:
            print("Warning: Could not determine DVC data hash for 'data/'. Ensure 'data/' is DVC-tracked and committed.")

    except Exception as e:
        print(f"Error getting DVC data hash: {e}. Is 'data/' DVC-tracked and DVC CLI accessible?")
        
    # Set MLflow experiment name and tracking URI
    # This makes sure Ultralytics knows where to log
    os.environ['MLFLOW_EXPERIMENT_NAME'] = 'YOLOv8_Vegetable_Counting'
    os.environ['MLFLOW_TRACKING_URI'] = 'mlruns' # Or 'http://localhost:5000' if running external server

    # 2. Load pre-trained YOLOv8 model
    model = YOLO(YOLOV8_MODEL_VARIANT)

    # 3. Train the model
    print(f"Starting training with {YOLOV8_MODEL_VARIANT} on {DATA_YAML_PATH} for {EPOCHS} epochs...")
    
    # Ultralytics will now automatically create and manage its own MLflow run.
    # It will log metrics and save artifacts (including best.pt) automatically.
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        # Set project and name for Ultralytics runs (helps organize in runs/ and MLflow)
        project='runs/detect',
        name=f'train_{YOLOV8_MODEL_VARIANT}_epochs{EPOCHS}'
    )

    # After training, you can optionally log additional custom metrics or artifacts
    # to the *active* MLflow run (which Ultralytics started and will eventually close).
    # To do this, you might need to get the current active run ID or let Ultralytics finish.
    # For simplicity, if you just want the DVC hash to be associated, you can use:
    if data_dvc_hash != "N/A":
        # Get the active run ID if Ultralytics has started one
        # This requires Ultralytics to have properly initiated the run.
        # A simpler way for MVP is to set the tag before starting train.
        # However, for Ultralytics's auto-MLflow, it's best to set env vars.
        # To add custom tags, it's best to use `mlflow.active_run()` after Ultralytics starts.
        # Ultralytics usually creates the run and then you can log to it.
        try:
            active_run = mlflow.active_run()
            if active_run:
                mlflow.set_tag("data_version", data_dvc_hash)
                print(f"Successfully logged DVC data hash {data_dvc_hash} to active MLflow run {active_run.info.run_id}")
            else:
                print("No active MLflow run found by `mlflow.active_run()` after Ultralytics train.")
        except Exception as e:
            print(f"Error logging DVC hash to MLflow: {e}")

    # The trained model is automatically saved by Ultralytics in `runs/detect/<name>/weights/best.pt`
    # and should also be logged by Ultralytics's MLflow integration.
    # You can access details from 'results' if needed.
    print(f"Training completed. Check MLflow UI for run details.")
    print(f"MLflow UI URL: {mlflow.get_tracking_uri()}")


if __name__ == '__main__':
    train_yolov8_model()