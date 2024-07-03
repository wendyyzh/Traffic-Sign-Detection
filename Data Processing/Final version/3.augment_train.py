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
annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_deleted.json'
image_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_del'
output_base_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_del'
new_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_augmented.json'

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

# Initialize dictionaries to count instances of each class and to store images by class
class_counts = {}
images_by_class = {}

# Resize function to reduce memory usage
def resize_image(image, max_size=(64, 64)):
    height, width = image.shape[:2]
    if height > max_size[0] or width > max_size[1]:
        scale = min(max_size[0] / height, max_size[1] / width)
        return cv2.resize(image, (int(width * scale), int(height * scale)))
    return image

# Iterate through the annotations and count instances for each class
for img_id, annotation in annotations.items():
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

        for obj in annotation['objects']:
            category = obj['category']
            if category in class_counts:
                class_counts[category] += 1
                images_by_class.setdefault(category, []).append((img_id, image))
            else:
                class_counts[category] = 1
                images_by_class[category] = [(img_id, image)]
    
    except cv2.error as e:
        logging.error(f"OpenCV error: {e}")

# Augment images to balance the classes with fewer than 500 instances
for category, images in images_by_class.items():
    if len(images) >= 500:
        logging.info(f"Category {category} already has {len(images)} instances. No augmentation needed.")
        continue
    elif len(images) >= 250:
        augment_needed = 500 - len(images)
        logging.info(f"Augmenting {category} with {augment_needed} images.")
        batch_size = 5  # Reduce batch size to save memory
        for start in range(0, augment_needed, batch_size):
            end = min(start + batch_size, augment_needed)
            batch_images = [img[1] for img in images[:end - start]]
            aug_images = augmenters(images=np.array(batch_images))

            # Save augmented images
            for i, aug_image in enumerate(aug_images):
                aug_img_id = f"{category}_aug_{start+i}"
                aug_img_path = os.path.join(output_base_path, f"{aug_img_id}.jpg")
                cv2.imwrite(aug_img_path, aug_image)
                annotations[aug_img_id] = {
                    'path': f"train/{aug_img_id}.jpg",
                    'objects': annotations.get(images[i][0], {}).get('objects', [])
                }

# Save the updated annotations to a new JSON file including both original and augmented images
new_data = {'imgs': annotations}
with open(new_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
