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
annotation_file_path = os.path.normpath(r'C:\Users\wezha\OneDrive\Desktop\tt100k_2021\tt100k_2021\annotations_all.json')
image_base_path = os.path.normpath(r'C:\Users\wezha\OneDrive\Desktop\tt100k_2021\tt100k_2021')  # Adjusted base path
output_base_path = os.path.normpath(r'C:\Users\wezha\OneDrive\Desktop\tt100k_2021\tt100k_2021\augmented_images')  # Output path for augmented images

# Load the JSON annotation file
with open(annotation_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Confirm the structure of the data
annotations = data.get('imgs')
if not annotations:
    raise ValueError("No 'imgs' key found in the data.")

# Define augmentation sequence
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flips
    iaa.Affine(rotate=(-30, 30)),  # Random rotations
    iaa.Affine(scale=(0.8, 1.2)),  # Random scaling
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Add Gaussian noise
])

# Create the output directory if it does not exist
os.makedirs(output_base_path, exist_ok=True)

# Initialize dictionaries to count instances of each class and to store images by class
class_counts = {}
images_by_class = {}

# Iterate through the annotations and count instances for each class
for img_id, annotation in annotations.items():
    image_path = os.path.normpath(os.path.join(image_base_path, annotation['path']))
    logging.info(f"Attempting to read image: {image_path}")
    
    # Debugging paths
    if not os.path.exists(image_path):
        logging.warning(f"Image file does not exist: {image_path}")
        continue
    
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Cannot read image: {image_path}")
        continue
    
    # Resize image if it is too large
    max_dimension = 1024
    if image.shape[0] > max_dimension or image.shape[1] > max_dimension:
        image = cv2.resize(image, (max_dimension, max_dimension))

    for obj in annotation['objects']:
        category = obj['category']
        if category in class_counts:
            class_counts[category] += 1
            images_by_class[category].append(image)
        else:
            class_counts[category] = 1
            images_by_class[category] = [image]

# Filter out classes with fewer than 500 instances
images_by_class = {k: v for k, v in images_by_class.items() if len(v) >= 500}
class_counts = {k: v for k, v in class_counts.items() if v >= 500}

# Augment images to balance the classes
augmented_class_counts = {key: len(value) for key, value in images_by_class.items()}
bounding_box_areas = []

for category, images in images_by_class.items():
    if len(images) < 500:
        continue
    elif len(images) >= 1000:
        logging.info(f"Category {category} already has {len(images)} instances. No augmentation needed.")
        continue

    augment_needed = 1000 - len(images)
    logging.info(f"Augmenting {category} with {augment_needed} images.")
    batch_size = 10
    for start in range(0, augment_needed, batch_size):
        end = min(start + batch_size, augment_needed)
        batch_images = images * (end - start)
        augmented_images = augmenters(images=np.array(batch_images))
        images_by_class[category].extend(augmented_images)
        augmented_class_counts[category] += len(augmented_images)

        # Save augmented images
        for i, aug_image in enumerate(augmented_images):
            output_image_path = os.path.normpath(os.path.join(output_base_path, f"{category}_aug_{start+i}.jpg"))
            cv2.imwrite(output_image_path, aug_image)

    # Clear the images from memory after processing
    images_by_class[category] = []

# Plot the histogram of class counts after augmentation
augmented_counts_df = pd.DataFrame(list(augmented_class_counts.items()), columns=['Class', 'Count'])
augmented_counts_df = augmented_counts_df.sort_values(by='Count', ascending=False)

plt.figure(figsize=(12, 8))
plt.bar(augmented_counts_df['Class'], augmented_counts_df['Count'])
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Number of Instances in Each Class After Augmentation')
plt.xticks(rotation=90)  # Rotate class labels for better readability
plt.show()

# Calculate bounding box areas for histogram (optional, can be omitted if not needed)
for img_id, annotation in annotations.items():
    for obj in annotation['objects']:
        bbox = obj['bbox']
        width = bbox['xmax'] - bbox['xmin']
        height = bbox['ymax'] - bbox['ymin']
        area = width * height
        bounding_box_areas.append(area)

# Define custom bins
bins = np.arange(0, 51000, 1000)  # Intervals of 1000 from 0 to 50,000
labels = [f"{i//1000}k-{(i+1000)//1000}k" for i in bins[:-2]] + ["50k"]

# Calculate histogram with custom bins
hist, bin_edges = np.histogram(bounding_box_areas, bins=bins)

# Plot the histogram of bounding box areas
plt.figure(figsize=(12, 8))
plt.bar(bin_edges[:-1], hist, width=1000, align='edge')
plt.xlabel('Bounding Box Area (pixels)')
plt.ylabel('Number of Instances')
plt.title('Histogram of Bounding Box Areas')
plt.xticks(bin_edges[:-1], labels, rotation=90)  # Rotate labels for better readability
plt.xlim(0, 50000)  # Set x-axis limit to 0 to 50,000
plt.show()
