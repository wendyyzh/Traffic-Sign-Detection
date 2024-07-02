import numpy as np
import json
import os
import shutil

train_dir = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_del'
train_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_augmented.json'
output_train_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_split.json'
output_val_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/val_split.json'
output_val_img_dir = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/val'
output_train_img_dir = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_split'

# Load the annotations
with open(train_annotation_file_path, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Confirm the structure of the data
train_annotations = annotations.get('imgs')
if not train_annotations:
    raise ValueError("No 'imgs' key found in the data.")

# Get the list of image IDs
img_ids = list(train_annotations.keys())

# Shuffle and split the image IDs
np.random.seed(1000)  # Fixed numpy random seed for reproducible shuffling
np.random.shuffle(img_ids)
split_point = int(len(img_ids) * 0.8)  # split at 80%

# Split into training and validation sets
train_ids = img_ids[:split_point]
val_ids = img_ids[split_point:]

# Create new annotation dictionaries
train_split_annotations = {'imgs': {img_id: train_annotations[img_id] for img_id in train_ids}}
val_split_annotations = {'imgs': {img_id: train_annotations[img_id] for img_id in val_ids}}

# Save the updated annotations to new JSON files
with open(output_train_file_path, 'w', encoding='utf-8') as f:
    json.dump(train_split_annotations, f, ensure_ascii=False, indent=4)

with open(output_val_file_path, 'w', encoding='utf-8') as f:
    json.dump(val_split_annotations, f, ensure_ascii=False, indent=4)

# Create the output directories for training and validation images if they don't exist
os.makedirs(output_val_img_dir, exist_ok=True)
os.makedirs(output_train_img_dir, exist_ok=True)

# Move training images to the training_split directory
for img_id in train_ids:
    img_info = train_annotations[img_id]
    relative_image_path = img_info['path']
    src_image_path = os.path.join(train_dir, os.path.basename(relative_image_path))
    dest_image_path = os.path.join(output_train_img_dir, os.path.basename(relative_image_path))
    if os.path.exists(src_image_path):
        shutil.move(src_image_path, dest_image_path)
    else:
        print(f"Image file does not exist: {src_image_path}")

# Move validation images to the validation directory
for img_id in val_ids:
    img_info = train_annotations[img_id]
    relative_image_path = img_info['path']
    src_image_path = os.path.join(train_dir, os.path.basename(relative_image_path))
    dest_image_path = os.path.join(output_val_img_dir, os.path.basename(relative_image_path))
    if os.path.exists(src_image_path):
        shutil.move(src_image_path, dest_image_path)
    else:
        print(f"Image file does not exist: {src_image_path}")

# Print the number of images in each set
print(f'Number of training images: {len(train_ids)}')
print(f'Number of validation images: {len(val_ids)}')