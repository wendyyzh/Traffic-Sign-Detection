import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from collections import defaultdict

# Load the JSON annotation file
annotation_file_path = '/Users/jessica_1/Documents/tt100k_2021/annotations_augmented.json'
with open(annotation_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Confirm the structure of the data
print(f"Data type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys in the data: {data.keys()}")
    annotations = data.get('imgs')
    if annotations:
        print(f"Annotations type: {type(annotations)}")
        if isinstance(annotations, dict) and len(annotations) > 0:
            first_key = next(iter(annotations))
            print(f"First annotation key: {first_key}")
            print(f"First annotation content: {annotations[first_key]}")
    else:
        print("No 'imgs' key found in the data.")
else:
    print("Data is not a dictionary.")

# Initialize dictionaries to count instances of each class
class_counts_train = defaultdict(int)
class_counts_augmented = defaultdict(int)
bounding_box_areas = []

# Count instances in the train folder
if annotations:
    for img_id, annotation in annotations.items():
        for obj in annotation['objects']:
            category = obj['category']
            class_counts_train[category] += 1
            
            # Calculate the bounding box area
            bbox = obj['bbox']
            width = bbox['xmax'] - bbox['xmin']
            height = bbox['ymax'] - bbox['ymin']
            area = width * height
            bounding_box_areas.append(area)

# Count instances in the augmented folder
augmented_image_base_path = '/Users/jessica_1/Documents/tt100k_2021/augmented/'
for image_file in os.listdir(augmented_image_base_path):
    if image_file.endswith('.jpg'):
        category = image_file.split('_')[0]
        class_counts_augmented[category] += 1

# Combine counts into a DataFrame
class_counts_combined = {
    'Class': [],
    'Train': [],
    'Augmented': []
}

all_classes = set(class_counts_train.keys()).union(set(class_counts_augmented.keys()))

for category in all_classes:
    class_counts_combined['Class'].append(category)
    class_counts_combined['Train'].append(class_counts_train.get(category, 0))
    class_counts_combined['Augmented'].append(class_counts_augmented.get(category, 0))

df_combined = pd.DataFrame(class_counts_combined)

# Plot the histogram
df_combined.set_index('Class').plot(kind='bar', stacked=True, figsize=(14, 8))
plt.xlabel('Class')
plt.ylabel('Number of Instances')
plt.title('Number of Instances per Class in Train and Augmented Datasets')
plt.xticks(rotation=90)  # Rotate class labels for better readability
plt.legend(title='Dataset')
plt.show()

# Plot bounding box area histogram
# Define custom bins
bins = np.arange(0, 51000, 1000)  # Intervals of 1000 from 0 to 50,000
labels = [f"{i//1000}-{(i+1000)//1000}" for i in bins[:-2]] + ["50"]

# Calculate histogram with custom bins
hist, bin_edges = np.histogram(bounding_box_areas, bins=bins)

# Plot the histogram of bounding box areas
plt.figure(figsize=(12, 8))
plt.bar(bin_edges[:-1], hist, width=1000, align='edge')
plt.xlabel('Bounding Box Area (in thousands of pixels)')
plt.ylabel('Number of Instances')
plt.title('Histogram of Bounding Box Areas')
plt.xticks(bin_edges[:-1], labels, rotation=90)  # Rotate labels for better readability
plt.xlim(0, 50000)  # Set x-axis limit to 0 to 50,000
plt.show()