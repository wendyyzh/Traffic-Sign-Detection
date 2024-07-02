import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Paths
annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/annotation_augment.json'
image_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train'
new_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_deleted.json'

# Load the JSON annotation file
with open(annotation_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Confirm the structure of the data
annotations = data.get('imgs')
if not annotations:
    raise ValueError("No 'imgs' key found in the data.")

# Initialize dictionaries to count instances of each class and track image usage
class_counts = {}
images_by_class = {}

# Iterate through the annotations and count instances for each class
for img_id, annotation in annotations.items():
    for obj in annotation['objects']:
        category = obj['category']
        if category in class_counts:
            class_counts[category] += 1
            images_by_class.setdefault(category, []).append(img_id)
        else:
            class_counts[category] = 1
            images_by_class[category] = [img_id]

# Identify classes with fewer than 250 instances
classes_to_delete = {k for k, v in class_counts.items() if v < 250}

# Remove annotations for classes with fewer than 250 instances and collect images to be checked for deletion
images_to_delete = set()
for category in classes_to_delete:
    for img_id in images_by_class[category]:
        if img_id in annotations:
            annotation = annotations[img_id]
            annotation['objects'] = [obj for obj in annotation['objects'] if obj['category'] != category]
            if not annotation['objects']:
                images_to_delete.add(img_id)
                annotations.pop(img_id, None)

# Verify if the images are used by any other classes with more than 250 instances
images_still_used = set()
for category, img_ids in images_by_class.items():
    if category not in classes_to_delete:
        for img_id in img_ids:
            if img_id in annotations:
                images_still_used.add(img_id)

# Remove images that are no longer used by any other class
for img_id in images_to_delete:
    if img_id not in images_still_used:
        image_path = os.path.join(image_base_path, os.path.basename(annotations.get(img_id, {}).get('path', '')))
        if os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"Deleted image: {image_path}")

# Save the updated annotations to a new JSON file
new_data = {'imgs': annotations}
with open(new_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

logging.info(f"Updated annotations saved to {new_annotation_file_path}")
