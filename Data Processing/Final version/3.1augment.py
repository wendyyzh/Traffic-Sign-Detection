import json
import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
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
    iaa.Multiply((0.8, 1.2)),  # Brightness change
])

# Initialize dictionaries to count instances of each class and to store images by class
class_counts = {}
images_by_class = {}

# Resize function to reduce memory usage
def resize_image(image, max_size=(640, 640)):
    height, width = image.shape[:2]
    scale = min(max_size[0] / height, max_size[1] / width)
    return cv2.resize(image, (int(width * scale), int(height * scale))), scale

# Clamp bounding boxes within image dimensions
def clamp_bbox(bb, img_shape):
    x1 = max(0, bb.x1)
    y1 = max(0, bb.y1)
    x2 = min(img_shape[1], bb.x2)
    y2 = min(img_shape[0], bb.y2)
    return BoundingBox(x1, y1, x2, y2)

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

        image, scale = resize_image(image)
        
        bbs = BoundingBoxesOnImage([
            BoundingBox(
                x1=obj['bbox']['xmin'] * scale, 
                y1=obj['bbox']['ymin'] * scale, 
                x2=obj['bbox']['xmax'] * scale, 
                y2=obj['bbox']['ymax'] * scale
            ) 
            for obj in annotation['objects']
        ], shape=image.shape)
        
        for obj in annotation['objects']:
            category = obj['category']
            if category in class_counts:
                class_counts[category] += 1
                images_by_class.setdefault(category, []).append((img_id, image, bbs))
            else:
                class_counts[category] = 1
                images_by_class[category] = [(img_id, image, bbs)]
    
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
            batch_bbs = [img[2] for img in images[:end - start]]
            aug_images, aug_bbs = augmenters(images=np.array(batch_images), bounding_boxes=batch_bbs)
            
            # Resize augmented images to 640x640 and adjust bounding boxes
            for i, (aug_image, aug_bb) in enumerate(zip(aug_images, aug_bbs)):
                aug_image, scale = resize_image(aug_image, max_size=(640, 640))
                aug_bb = aug_bb.on(aug_image)
                aug_bb = [clamp_bbox(bb, aug_image.shape) for bb in aug_bb]
                
                aug_img_id = f"{category}_aug_{start+i}"
                aug_img_path = os.path.join(output_base_path, f"{aug_img_id}.jpg")
                cv2.imwrite(aug_img_path, aug_image)
                
                new_objects = [{
                    "bbox": {
                        "xmin": bb.x1 / scale,
                        "ymin": bb.y1 / scale,
                        "xmax": bb.x2 / scale,
                        "ymax": bb.y2 / scale
                    },
                    "category": obj['category']
                } for bb, obj in zip(aug_bb, annotation['objects'])]
                
                annotations[aug_img_id] = {
                    'path': f"train/{aug_img_id}.jpg",
                    'objects': new_objects
                }

# Save the updated annotations to a new JSON file including both original and augmented images
new_data = {'imgs': annotations}
with open(new_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

