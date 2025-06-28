import os
from collections import defaultdict

# --- Configuration ---
# Base directory where you unzipped CVAT exports (e.g., 'cvat_exports/annotations_v1')
CVAT_EXPORTS_BASE_DIR = 'cvat_exports/annotations_v1'

# The order of your classes MUST match the class_id (0, 1, 2...) in your .txt files
# This list should be identical to the CLASS_NAMES in your prepare_dataset.py and data.yaml
CLASS_NAMES = ['carrot', 'bean', 'radish']
# Add other classes here as you annotate them, maintaining their correct 0-indexed order.

# --- Main Counting Logic ---
def count_annotations_per_class_in_cvat_splits():
    # Initialize counters for each split
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    total_train_objects = 0
    total_val_objects = 0
    
    # Define paths for CVAT exported labels
    cvat_train_labels_dir = os.path.join(CVAT_EXPORTS_BASE_DIR, 'labels', 'train')
    cvat_val_labels_dir = os.path.join(CVAT_EXPORTS_BASE_DIR, 'labels', 'validation') # or 'val' if CVAT exports as 'val'

    # --- Process TRAIN split annotations ---
    if os.path.exists(cvat_train_labels_dir):
        print(f"Scanning train annotations in: {cvat_train_labels_dir}")
        train_label_files = [f for f in os.listdir(cvat_train_labels_dir) if f.endswith('.txt')]
        
        if not train_label_files:
            print("No .txt annotation files found in train split.")
        
        for label_file in train_label_files:
            file_path = os.path.join(cvat_train_labels_dir, label_file)
            objects_in_file = process_single_label_file(file_path, CLASS_NAMES, train_counts)
            total_train_objects += objects_in_file
    else:
        print(f"Warning: CVAT train labels directory not found: {cvat_train_labels_dir}. No train annotations counted.")

    # --- Process VALIDATION split annotations ---
    if os.path.exists(cvat_val_labels_dir):
        print(f"Scanning validation annotations in: {cvat_val_labels_dir}")
        val_label_files = [f for f in os.listdir(cvat_val_labels_dir) if f.endswith('.txt')]

        if not val_label_files:
            print("No .txt annotation files found in validation split.")

        for label_file in val_label_files:
            file_path = os.path.join(cvat_val_labels_dir, label_file)
            objects_in_file = process_single_label_file(file_path, CLASS_NAMES, val_counts)
            total_val_objects += objects_in_file
    else:
        print(f"Warning: CVAT validation labels directory not found: {cvat_val_labels_dir}. No validation annotations counted.")

    # --- Summary Report ---
    print(f"\n--- Annotation Summary Across Splits ---")
    
    # Train Summary
    print(f"\nTrain Split (Total Objects: {total_train_objects}):")
    if total_train_objects == 0:
        print("  No objects found in the training annotations.")
    else:
        for class_name in CLASS_NAMES:
            print(f"  {class_name} (ID {CLASS_NAMES.index(class_name)}): {train_counts[class_name]} objects")

    # Validation Summary
    print(f"\nValidation Split (Total Objects: {total_val_objects}):")
    if total_val_objects == 0:
        print("  No objects found in the validation annotations.")
    else:
        for class_name in CLASS_NAMES:
            print(f"  {class_name} (ID {CLASS_NAMES.index(class_name)}): {val_counts[class_name]} objects")

    # Overall Summary
    total_objects_overall = total_train_objects + total_val_objects
    print(f"\nOverall Total Objects: {total_objects_overall}")
    if total_objects_overall > 0:
        overall_counts = defaultdict(int)
        for class_name in CLASS_NAMES:
            overall_counts[class_name] = train_counts[class_name] + val_counts[class_name]
            print(f"  {class_name} (ID {CLASS_NAMES.index(class_name)}): {overall_counts[class_name]} objects")

    # Provide guidance based on counts (example, adjust targets)
    TARGET_INSTANCES_PER_CLASS = 50 # Example target
    print(f"\n--- Annotation Progress ---")
    all_targets_met = True
    for class_name in CLASS_NAMES:
        overall_count = overall_counts[class_name] # Using overall_counts from above
        if overall_count < TARGET_INSTANCES_PER_CLASS:
            all_targets_met = False
            print(f"  Needs more {class_name}: Currently {overall_count}, target {TARGET_INSTANCES_PER_CLASS}.")
        else:
            print(f"  {class_name} target met: {overall_count} / {TARGET_INSTANCES_PER_CLASS} objects.")
    
    if all_targets_met:
        print("\nAll target instance counts have been met!")
    else:
        print("\nContinue annotating to meet targets for all classes.")


# --- Helper function to process a single label file ---
def process_single_label_file(file_path, class_names_list, counts_dict):
    """Reads a single YOLO .txt file, updates counts_dict, and returns objects found in file."""
    objects_in_file = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5: # Ensure it's a valid YOLO annotation line
                    class_id = int(parts[0])
                    if 0 <= class_id < len(class_names_list):
                        counts_dict[class_names_list[class_id]] += 1
                        objects_in_file += 1
                    else:
                        print(f"Warning: Class ID {class_id} in {os.path.basename(file_path)} is out of bounds for defined CLASS_NAMES. Skipping object count.")
                else:
                    print(f"Warning: Malformed line in {os.path.basename(file_path)}: '{line.strip()}'. Skipping line.")
    except Exception as e:
        print(f"Error reading label file {os.path.basename(file_path)}: {e}")
    return objects_in_file

if __name__ == '__main__':
    # When running from inventory_counter/ project root:
    # This script would typically be in src/count_annotations.py
    count_annotations_per_class_in_cvat_splits()