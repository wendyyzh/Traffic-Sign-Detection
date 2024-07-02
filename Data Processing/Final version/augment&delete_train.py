import json
import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Paths
annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train.json'
image_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train'
output_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/aug_train_image'
new_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/annotation_augment.json'

# Load the JSON annotation file
with open(annotation_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Confirm the structure of the data
annotations = data.get('imgs')
if not annotations:
    raise ValueError("No 'imgs' key found in the data.")

# Define augmentation sequence
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-30, 30)),
    iaa.Affine(scale=(0.8, 1.2)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
])

# Create the output directory if it does not exist
os.makedirs(output_base_path, exist_ok=True)

# Initialize dictionaries to count instances of each class and to store images by class
class_counts = {}
images_by_class = {}

# Track deleted images
deleted_images = set()

# Resize function to reduce memory usage
def resize_image(image, max_size=(64, 64)):
    height, width = image.shape[:2]
    if height > max_size[0] or width > max_size[1]:
        scale = min(max_size[0] / height, max_size[1] / width)
        return cv2.resize(image, (int(width * scale), int(height * scale)))
    return image

# Iterate through the annotations and count instances for each class
for img_id, annotation in annotations.items():
    # Correct the path to avoid duplicate "train" directory in the path
    image_path = os.path.join(image_base_path, os.path.basename(annotation.get('path', '')))
    logging.info(f"Attempting to read image: {image_path}")
    if not os.path.exists(image_path):
        logging.warning(f"Image file does not exist: {image_path}")
        continue

    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Cannot read image: {image_path}")
            continue

        image = resize_image(image)

        valid_objects = []
        for obj in annotation['objects']:
            category = obj['category']
            if category in class_counts:
                class_counts[category] += 1
                images_by_class.setdefault(category, []).append((img_id, image))
                valid_objects.append(obj)
            else:
                class_counts[category] = 1
                images_by_class[category] = [(img_id, image)]
                valid_objects.append(obj)
        
        if not valid_objects:
            deleted_images.add(img_id)
            logging.info(f"Deleting image {img_id} because all instances are in classes with fewer than 250 images.")
    
    except cv2.error as e:
        logging.error(f"OpenCV error: {e}")

# Identify classes with fewer than 250 instances and remove their annotations
classes_to_delete = {k for k, v in class_counts.items() if v < 250}

for category in classes_to_delete:
    for img_id, _ in images_by_class.get(category, []):
        if img_id in annotations:
            annotation = annotations[img_id]
            annotation['objects'] = [obj for obj in annotation['objects'] if obj['category'] != category]
            if not annotation['objects']:
                image_path = os.path.join(image_base_path, os.path.basename(annotation.get('path', '')))
                if os.path.exists(image_path):
                    os.remove(image_path)
                annotations.pop(img_id, None)
                deleted_images.add(img_id)

# Augment images to balance the classes with fewer than 500 instances
new_annotations = {}

for category, images in images_by_class.items():
    if len(images) >= 500:
        logging.info(f"Category {category} already has {len(images)} instances. No augmentation needed.")
        for img_id, image in images:
            if img_id in annotations and img_id not in deleted_images:
                new_annotations[img_id] = annotations[img_id]
        continue
    elif len(images) >= 250:
        augment_needed = 500 - len(images)
        logging.info(f"Augmenting {category} with {augment_needed} images.")
        batch_size = 5  # Reduce batch size to save memory
        augmented_images = []
        for start in range(0, augment_needed, batch_size):
            end = min(start + batch_size, augment_needed)
            batch_images = [img[1] for img in images[:end - start]]
            aug_images = augmenters(images=np.array(batch_images))
            augmented_images.extend(aug_images)

            # Save augmented images
            for i, aug_image in enumerate(aug_images):
                if len(images_by_class[category]) >= 500:
                    break
                aug_img_id = f"{category}_aug_{start+i}"
                aug_img_path = os.path.join(output_base_path, f"{category}_aug_{start+i}.jpg")
                cv2.imwrite(aug_img_path, aug_image)
                new_annotations[aug_img_id] = {
                    'path': f"aug_train_image/{category}_aug_{start+i}.jpg",
                    'objects': annotations.get(images[i][0], {}).get('objects', [])
                }
                images_by_class[category].append((aug_img_id, aug_image))

# Save new annotations to a JSON file containing only augmented and non-deleted original annotations
new_data = {'imgs': new_annotations}
with open(new_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

# Plot the histogram of class counts after augmentation
augmented_class_counts = {key: len(value) for key, value in images_by_class.items()}
augmented_counts_df = pd.DataFrame(list(augmented_class_counts.items()), columns=['Class', 'Count'])
augmented_counts_df = augmented_counts_df.sort_values(by='Count', ascending=False)

plt.figure(figsize=(12, 8))
plt.bar(augmented_counts_df['Class'], augmented_counts_df['Count'])
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Number of Instances in Each Class After Augmentation')
plt.xticks(rotation=90)  # Rotate class labels for better readability
plt.show()