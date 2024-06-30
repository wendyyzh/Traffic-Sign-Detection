import json
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Paths
annotation_file_path = os.path.normpath(r'C:\Users\wezha\OneDrive\Desktop\tt100k_2021\tt100k_2021\annotations_all.json')
image_base_path = os.path.normpath(r'C:\Users\wezha\OneDrive\Desktop\tt100k_2021\tt100k_2021')

# Load the JSON annotation file
with open(annotation_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Confirm the structure of the data
annotations = data.get('imgs')
if not annotations:
    raise ValueError("No 'imgs' key found in the data.")

# Initialize dictionaries to count instances of each class and to store images by class
class_counts = {}
images_by_class = {}

# Iterate through the annotations and count instances for each class
for img_id, annotation in annotations.items():
    image_path = os.path.normpath(os.path.join(image_base_path, annotation['path']))
    logging.info(f"Attempting to read image: {image_path}")
    
    if not os.path.exists(image_path):
        logging.warning(f"Image file does not exist: {image_path}")
        continue
    
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Cannot read image: {image_path}")
        continue
    
    for obj in annotation['objects']:
        category = obj['category']
        if category in class_counts:
            class_counts[category] += 1
            images_by_class[category].append(image_path)
        else:
            class_counts[category] = 1
            images_by_class[category] = [image_path]

# Filter out and delete images for classes with fewer than 500 instances
for category, image_paths in list(images_by_class.items()):
    if len(image_paths) < 500:
        logging.info(f"Deleting images for category {category} with {len(image_paths)} instances (less than 500).")
        for image_path in image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)
        del images_by_class[category]

# Plot the histogram of class counts after deletion
filtered_class_counts = {k: len(v) for k, v in images_by_class.items()}
filtered_counts_df = pd.DataFrame(list(filtered_class_counts.items()), columns=['Class', 'Count'])
filtered_counts_df = filtered_counts_df.sort_values(by='Count', ascending=False)

plt.figure(figsize=(12, 8))
plt.bar(filtered_counts_df['Class'], filtered_counts_df['Count'])
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Number of Instances in Each Class After Deletion')
plt.xticks(rotation=90)  # Rotate class labels for better readability
plt.show()