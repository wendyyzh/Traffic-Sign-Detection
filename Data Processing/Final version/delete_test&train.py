import json
import os
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

# Remove annotations for classes with fewer than 250 instances and collect images to be checked for deletion
images_to_delete = set()
for category in classes_to_delete:
    for img_id in images_by_class[category]:
        if img_id in train_annotations:
            annotation = train_annotations[img_id]
            annotation['objects'] = [obj for obj in annotation['objects'] if obj['category'] != category]
            if not annotation['objects']:
                images_to_delete.add(img_id)
                train_annotations.pop(img_id, None)

# Verify if the images are used by any other classes with more than 250 instances
images_still_used = set()
for category, img_ids in images_by_class.items():
    if category not in classes_to_delete:
        for img_id in img_ids:
            if img_id in train_annotations:
                images_still_used.add(img_id)

# Remove images that are no longer used by any other class
for img_id in list(images_to_delete):  # Iterate over a copy of the set
    if img_id not in images_still_used:
        image_info = train_annotations.get(img_id, {})
        if 'path' in image_info and image_info['path']:
            relative_image_path = image_info['path']
            image_path = os.path.join(train_image_base_path, relative_image_path)
            if os.path.exists(image_path) and os.path.isfile(image_path):
                try:
                    os.remove(image_path)
                    logging.info(f"Deleted image: {image_path}")
                except PermissionError as e:
                    logging.error(f"PermissionError: {e} - {image_path}")
                except Exception as e:
                    logging.error(f"Error: {e} - {image_path}")

# Save the updated train annotations to a new JSON file
new_train_data = {'imgs': train_annotations}
with open(train_deleted_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)

logging.info(f"Updated train annotations saved to {train_deleted_annotation_file_path}")

# Process the test annotations to delete classes no longer in train_deleted.json
with open(train_deleted_annotation_file_path, 'r', encoding='utf-8') as f:
    train_deleted_data = json.load(f)

train_deleted_annotations = train_deleted_data.get('imgs')
valid_classes = set()

for img_id, annotation in train_deleted_annotations.items():
    for obj in annotation['objects']:
        valid_classes.add(obj['category'])

# Remove annotations in test.json if the class is no longer in train_deleted.json
test_images_to_delete = set()
for img_id in list(test_annotations.keys()):  # Iterate over a copy of the dictionary keys
    annotation = test_annotations[img_id]
    annotation['objects'] = [obj for obj in annotation['objects'] if obj['category'] in valid_classes]
    if not annotation['objects']:
        test_images_to_delete.add(img_id)
        test_annotations.pop(img_id)

# Remove test images that only include the deleted classes
for img_id in test_images_to_delete:
    image_info = test_annotations.get(img_id, {})
    if 'path' in image_info and image_info['path']:
        relative_image_path = image_info['path']
        image_path = os.path.join(test_image_base_path, relative_image_path)
        if os.path.exists(image_path) and os.path.isfile(image_path):
            try:
                os.remove(image_path)
                logging.info(f"Deleted test image: {image_path}")
            except PermissionError as e:
                logging.error(f"PermissionError: {e} - {image_path}")
            except Exception as e:
                logging.error(f"Error: {e} - {image_path}")

# Save the updated test annotations to a new JSON file
new_test_data = {'imgs': test_annotations}
with open(test_deleted_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_test_data, f, ensure_ascii=False, indent=4)

logging.info(f"Updated test annotations saved to {test_deleted_annotation_file_path}")
