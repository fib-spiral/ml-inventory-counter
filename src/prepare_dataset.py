import os
import shutil
import yaml
from collections import defaultdict

# --- Configuration ---
RAW_IMAGES_DIR = 'raw_images'
CVAT_EXPORTS_BASE_DIR = 'cvat_exports/annotations_v1' # Base of CVAT export, e.g., where 'labels/' is
OUTPUT_DATA_DIR = 'data' # The final, prepared dataset for YOLO

# Define your FINAL, DESIRED class names in their 0-indexed order.
# This list MUST match the class_id (0, 1, 2...) in your exported YOLO .txt files.
# Example: If 'carrot' is class_id 0, 'bean' is 1, 'radish' is 2 in your .txt files.
CLASS_NAMES = ['carrot', 'bean', 'radish']
# Add other classes here as you annotate them, maintaining their correct 0-indexed order.

# --- Main Logic ---
def prepare_dataset():
    # 1. Clean up/create output directories
    if os.path.exists(OUTPUT_DATA_DIR):
        print(f"Cleaning existing output directory: {OUTPUT_DATA_DIR}")
        shutil.rmtree(OUTPUT_DATA_DIR)

    os.makedirs(os.path.join(OUTPUT_DATA_DIR, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATA_DIR, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATA_DIR, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATA_DIR, 'val', 'labels'), exist_ok=True)
    # No test directory creation for now

    # Load all unique image basenames from raw_images/
    # (e.g., 'carrot_train_001' from 'carrot_train_001.jpg')
    all_raw_images_basenames = {os.path.splitext(f)[0] for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    raw_images_full_paths = {os.path.splitext(f)[0]: os.path.join(RAW_IMAGES_DIR, f) for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    print(f"Found {len(all_raw_images_basenames)} images in {RAW_IMAGES_DIR}")

    # Initialize counters for images and object instances
    train_image_count = 0
    val_image_count = 0
    skipped_image_count = 0
    total_objects_per_class = defaultdict(int)

    # Define paths for CVAT exported labels (assuming 'labels/train' and 'labels/validation' structure)
    cvat_train_labels_dir = os.path.join(CVAT_EXPORTS_BASE_DIR, 'labels', 'train')
    cvat_val_labels_dir = os.path.join(CVAT_EXPORTS_BASE_DIR, 'labels', 'validation') # or 'val' if CVAT exports as 'val'

    # Check if export directories exist
    if not os.path.exists(cvat_train_labels_dir):
        print(f"Warning: CVAT train labels directory not found: {cvat_train_labels_dir}")
    if not os.path.exists(cvat_val_labels_dir):
        print(f"Warning: CVAT validation labels directory not found: {cvat_val_labels_dir}")

    # --- Process annotated images from CVAT's TRAIN split ---
    if os.path.exists(cvat_train_labels_dir):
        for label_filename in os.listdir(cvat_train_labels_dir):
            if not label_filename.endswith('.txt'): continue

            base_name = os.path.splitext(label_filename)[0]
            if base_name not in all_raw_images_basenames:
                print(f"Warning: Label '{label_filename}' in CVAT train export has no corresponding image in '{RAW_IMAGES_DIR}'. Skipping.")
                skipped_image_count += 1
                continue
            
            raw_img_path = raw_images_full_paths[base_name]
            exported_label_path = os.path.join(cvat_train_labels_dir, label_filename)

            target_img_dir = os.path.join(OUTPUT_DATA_DIR, 'train', 'images')
            target_lbl_dir = os.path.join(OUTPUT_DATA_DIR, 'train', 'labels')
            
            if copy_and_count_annotation(raw_img_path, exported_label_path, target_img_dir, target_lbl_dir,
                                         CLASS_NAMES, total_objects_per_class):
                train_image_count += 1
            else:
                skipped_image_count += 1 # Error during processing, so it's skipped


    # --- Process annotated images from CVAT's VALIDATION split ---
    if os.path.exists(cvat_val_labels_dir):
        for label_filename in os.listdir(cvat_val_labels_dir):
            if not label_filename.endswith('.txt'): continue

            base_name = os.path.splitext(label_filename)[0]
            if base_name not in all_raw_images_basenames:
                print(f"Warning: Label '{label_filename}' in CVAT val export has no corresponding image in '{RAW_IMAGES_DIR}'. Skipping.")
                skipped_image_count += 1
                continue

            raw_img_path = raw_images_full_paths[base_name]
            exported_label_path = os.path.join(cvat_val_labels_dir, label_filename)

            target_img_dir = os.path.join(OUTPUT_DATA_DIR, 'val', 'images')
            target_lbl_dir = os.path.join(OUTPUT_DATA_DIR, 'val', 'labels')

            if copy_and_count_annotation(raw_img_path, exported_label_path, target_img_dir, target_lbl_dir,
                                         CLASS_NAMES, total_objects_per_class):
                val_image_count += 1
            else:
                skipped_image_count += 1 # Error during processing

    print(f"\n--- Dataset Preparation Summary ---")
    print(f"Annotated and prepared: {train_image_count} training images, {val_image_count} validation images.")
    print(f"Total objects per class:")
    for class_name in CLASS_NAMES: # Use the defined CLASS_NAMES for consistent output order
        print(f"  {class_name} (ID {CLASS_NAMES.index(class_name)}): {total_objects_per_class[class_name]} objects")
    
    print(f"Skipped {skipped_image_count} images (missing from raw_images or error during processing).")
    
    if train_image_count + val_image_count == 0:
        print("No annotated images were processed. 'data/' directory is empty.")
        return # Exit if no data was processed

    # 2. Create data.yaml for YOLO with the defined CLASS_NAMES
    data_yaml_content = {
        'path': os.path.abspath(OUTPUT_DATA_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }

    with open(os.path.join(OUTPUT_DATA_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    
    print(f"data.yaml created at '{os.path.join(OUTPUT_DATA_DIR, 'data.yaml')}'")
    print(f"Dataset prepared and saved to '{OUTPUT_DATA_DIR}'")

# --- Helper function for copying and counting ---
def copy_and_count_annotation(raw_img_path, exported_label_path, target_img_dir, target_lbl_dir,
                              class_names_list, total_objects_per_class_dict):
    """
    Copies an image and its label file to the target directories,
    and counts objects based on the provided class_names_list.
    """
    try:
        # Read the label file to count objects
        with open(exported_label_path, 'r') as f_in:
            lines = f_in.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Warning: Malformed line in {os.path.basename(exported_label_path)}: '{line.strip()}'")
                    continue

                class_id = int(parts[0])
                if 0 <= class_id < len(class_names_list):
                    total_objects_per_class_dict[class_names_list[class_id]] += 1
                else:
                    print(f"Warning: Class ID {class_id} in {os.path.basename(exported_label_path)} is out of bounds for defined CLASS_NAMES. Skipping object count.")
        
        # Copy the label file
        shutil.copy(exported_label_path, os.path.join(target_lbl_dir, os.path.basename(exported_label_path)))
        
        # Copy the corresponding image
        shutil.copy(raw_img_path, os.path.join(target_img_dir, os.path.basename(raw_img_path)))
        return True # Successfully processed
    except Exception as e:
        print(f"Error processing {os.path.basename(exported_label_path)}: {e}")
        return False # Failed to process


if __name__ == '__main__':
    prepare_dataset()