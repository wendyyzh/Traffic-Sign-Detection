import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def check_permissions(directory):
    test_file_path = os.path.join(directory, 'test_permission.txt')
    
    try:
        # Attempt to create a test file
        with open(test_file_path, 'w') as test_file:
            test_file.write('Testing permissions.')
        
        # Attempt to read the test file
        with open(test_file_path, 'r') as test_file:
            content = test_file.read()
            if content != 'Testing permissions.':
                raise PermissionError("Read operation failed.")
        
        # Attempt to delete the test file
        os.remove(test_file_path)
        
        logging.info(f"Permissions are sufficient for directory: {directory}")
        return True
    except PermissionError as e:
        logging.error(f"PermissionError: {e}")
    except Exception as e:
        logging.error(f"Error: {e}")
    
    logging.error(f"Insufficient permissions for directory: {directory}")
    return False

# Paths
annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train.json'
image_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train'
new_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_deleted.json'

# Check permissions
if not check_permissions(image_base_path):
    raise PermissionError(f"Insufficient permissions for directory: {image_base_path}")

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
        image_info = annotations.get(img_id, {})
        image_path = os.path.join(image_base_path, os.path.basename(image_info.get('path', '')))
        if os.path.exists(image_path):
            if os.path.isfile(image_path):  # Check if it's a file
                try:
                    os.remove(image_path)
                    logging.info(f"Deleted image: {image_path}")
                except PermissionError as e:
                    logging.error(f"PermissionError: {e} - {image_path}")
                except Exception as e:
                    logging.error(f"Error: {e} - {image_path}")
            else:
                logging.warning(f"Path is not a file: {image_path}")

# Save the updated annotations to a new JSON file
new_data = {'imgs': annotations}
with open(new_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

logging.info(f"Updated annotations saved to {new_annotation_file_path}")
