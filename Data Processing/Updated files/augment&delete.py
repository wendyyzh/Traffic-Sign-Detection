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
annotation_file_path = '/Users/jessica_1/Documents/tt100k_2021/annotations_all.json'
image_base_path = '/Users/jessica_1/Documents/tt100k_2021/'  # Adjusted base path
output_base_path = '/Users/jessica_1/Documents/tt100k_2021/augmented/'  # Output path for augmented images
new_annotation_file_path = '/Users/jessica_1/Documents/tt100k_2021/annotations_augmented.json'  # Path for new annotation file

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

# Resize function to reduce memory usage
def resize_image(image, max_size=(64, 64)):
    height, width = image.shape[:2]
    if height > max_size[0] or width > max_size[1]:
        scale = min(max_size[0] / height, max_size[1] / width)
        return cv2.resize(image, (int(width * scale), int(height * scale)))
    return image

# Iterate through the annotations and count instances for each class
for img_id, annotation in annotations.items():
    image_path = os.path.join(image_base_path, annotation.get('path', ''))
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
                images_by_class[category].append((img_id, image))
            else:
                class_counts[category] = 1
                images_by_class[category] = [(img_id, image)]
    except cv2.error as e:
        logging.error(f"OpenCV error: {e}")

# Identify classes with fewer than 500 instances
classes_to_delete = {k for k, v in class_counts.items() if v < 500}

# Delete images belonging to these classes
for category in classes_to_delete:
    for img_id, image in images_by_class[category]:
        # Check if the image is used for other classes
        other_classes_using_image = [
            other_category for other_category, imgs in images_by_class.items()
            if other_category != category and any(img[0] == img_id for img in imgs)
        ]

        # If no other classes are using this image, proceed to delete
        if not other_classes_using_image:
            if img_id in annotations:
                image_path = os.path.join(image_base_path, annotations[img_id].get('path', ''))
                if os.path.exists(image_path):
                    os.remove(image_path)
                annotations.pop(img_id, None)
        else:
            logging.info(f"Image {img_id} used by other classes: {other_classes_using_image}")

    images_by_class.pop(category, None)
    class_counts.pop(category, None)

# Augment images to balance the classes
augmented_class_counts = {key: len(value) for key, value in images_by_class.items()}
bounding_box_areas = []
new_annotations = {}

for category, images in images_by_class.items():
    if len(images) >= 1000:
        logging.info(f"Category {category} already has {len(images)} instances. No augmentation needed.")
        for img_id, image in images:
            if img_id in annotations:
                new_annotations[img_id] = annotations[img_id]
        continue

    augment_needed = 1000 - len(images)
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
            if len(images_by_class[category]) >= 1000:
                break
            aug_img_id = f"{category}_aug_{start+i}"
            aug_img_path = f"{output_base_path}{category}_aug_{start+i}.jpg"
            cv2.imwrite(aug_img_path, aug_image)
            new_annotations[aug_img_id] = {
                'path': f"augmented/{category}_aug_{start+i}.jpg",
                'objects': annotations.get(images[i][0], {}).get('objects', [])
            }
            images_by_class[category].append((aug_img_id, aug_image))

    augmented_class_counts[category] = len(images_by_class[category])

    # Add original images to the new annotations
    for img_id, image in images:
        if img_id in annotations:
            new_annotations[img_id] = annotations[img_id]

# Save new annotations to a JSON file
new_data = {'imgs': new_annotations}
with open(new_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

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
for img_id, annotation in new_annotations.items():
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
