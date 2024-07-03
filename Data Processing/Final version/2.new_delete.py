import json
import os
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Paths
train_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train.json'
train_deleted_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_deleted.json'
test_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test.json'
test_deleted_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test_deleted.json'
train_image_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021'
test_image_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021'
train_del_image_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_del'
test_del_image_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test_del'

# Create new directories if they don't exist
os.makedirs(train_del_image_base_path, exist_ok=True)
os.makedirs(test_del_image_base_path, exist_ok=True)

# Load the JSON annotation files
with open(train_annotation_file_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open(test_annotation_file_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Confirm the structure of the data
train_annotations = train_data.get('imgs')
test_annotations = test_data.get('imgs')
if not train_annotations or not test_annotations:
    raise ValueError("No 'imgs' key found in one or both of the data files.")

# Initialize dictionaries to count instances of each class and track image usage
class_counts = {}
images_by_class = {}

# Iterate through the train annotations and count instances for each class
for img_id, annotation in train_annotations.items():
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
valid_classes = set(class_counts.keys()) - classes_to_delete

# Create the "types" list for classes with more than 250 instances
types_list = list(valid_classes)

# Remove annotations for classes with fewer than 250 instances and collect images to be moved to the new folder
images_to_delete = set()
for img_id, annotation in list(train_annotations.items()):
    annotation['objects'] = [obj for obj in annotation['objects'] if obj['category'] in valid_classes]
    if not annotation['objects']:
        images_to_delete.add(img_id)
        train_annotations.pop(img_id, None)

# Move the images that are still used to the new train_del folder
for img_id, annotation in train_annotations.items():
    image_info = annotation
    if 'path' in image_info and image_info['path']:
        relative_image_path = image_info['path']
        original_image_path = os.path.join(train_image_base_path, relative_image_path)
        new_image_path = os.path.join(train_del_image_base_path, os.path.basename(relative_image_path))
        if os.path.exists(original_image_path) and os.path.isfile(original_image_path):
            try:
                shutil.move(original_image_path, new_image_path)
                logging.info(f"Moved image to train_del: {new_image_path}")
            except Exception as e:
                logging.error(f"Error moving image: {e} - {original_image_path}")
        else:
            logging.warning(f"Image does not exist or is not a file: {original_image_path}")

# Save the updated train annotations to a new JSON file
new_train_data = {'types': types_list, 'imgs': train_annotations}
with open(train_deleted_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)

logging.info(f"Updated train annotations saved to {train_deleted_annotation_file_path}")

# Process the test annotations to delete classes no longer in train_deleted.json
with open(train_deleted_annotation_file_path, 'r', encoding='utf-8') as f:
    train_deleted_data = json.load(f)

train_deleted_annotations = train_deleted_data.get('imgs')

# Remove annotations in test.json if the class is no longer in train_deleted.json
test_images_to_delete = set()
for img_id in list(test_annotations.keys()):  # Iterate over a copy of the dictionary keys
    annotation = test_annotations[img_id]
    annotation['objects'] = [obj for obj in annotation['objects'] if obj['category'] in valid_classes]
    if not annotation['objects']:
        test_images_to_delete.add(img_id)
        test_annotations.pop(img_id)

# Move the test images that are still used to the new test_del folder
for img_id, annotation in test_annotations.items():
    image_info = annotation
    if 'path' in image_info and image_info['path']:
        relative_image_path = image_info['path']
        original_image_path = os.path.join(test_image_base_path, relative_image_path)
        new_image_path = os.path.join(test_del_image_base_path, os.path.basename(relative_image_path))
        if os.path.exists(original_image_path) and os.path.isfile(original_image_path):
            try:
                shutil.move(original_image_path, new_image_path)
                logging.info(f"Moved image to test_del: {new_image_path}")
            except Exception as e:
                logging.error(f"Error moving image: {e} - {original_image_path}")
        else:
            logging.warning(f"Image does not exist or is not a file: {original_image_path}")

# Save the updated test annotations to a new JSON file
new_test_data = {'types': types_list, 'imgs': test_annotations}
with open(test_deleted_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_test_data, f, ensure_ascii=False, indent=4)

logging.info(f"Updated test annotations saved to {test_deleted_annotation_file_path}")
