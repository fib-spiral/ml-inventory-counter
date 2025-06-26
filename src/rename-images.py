import os
import shutil

# Define your base raw data path
base_raw_path = 'data/raw/' # This is where your current 'test' and 'train' folders are

# Define the target flattened raw images directory
target_raw_images_dir = 'raw_images/'

# Create the target directory if it doesn't exist
os.makedirs(target_raw_images_dir, exist_ok=True)

# Iterate through 'train' and 'test' folders
for split_type in ['train', 'test', 'validation']:
    split_path = os.path.join(base_raw_path, split_type)

    # Iterate through each vegetable category (e.g., 'carrot', 'bean')
    for category_name in os.listdir(split_path):
        category_path = os.path.join(split_path, category_name)

        # Ensure it's a directory
        if os.path.isdir(category_path):
            print(f"Processing category: {category_name} in split: {split_type}")
            # Iterate through images in the category folder
            for filename in os.listdir(category_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')): # Only process image files
                    old_filepath = os.path.join(category_path, filename)
                    # Create a new unique filename
                    # Format: category_split_originalfilename.ext (e.g., carrot_train_image_01.jpg)
                    new_filename = f"{category_name}_{split_type}_{filename}"
                    new_filepath = os.path.join(target_raw_images_dir, new_filename)

                    # Copy (or move) the file to the new location
                    shutil.copy(old_filepath, new_filepath) # Use shutil.move if you want to delete originals

print(f"\nAll raw images flattened and renamed to '{target_raw_images_dir}'")