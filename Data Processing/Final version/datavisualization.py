import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the JSON annotation file
annotation_file_path ='C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test.json'
with open(annotation_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Confirm the structure of the data
print(f"Data type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys in the data: {data.keys()}")
    # Annotations are likely under the 'imgs' key
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

# Initialize a dictionary to count instances of each class
class_counts = {}
bounding_box_areas = []

# Iterate through the annotations and count instances for each class
if annotations:
    for img_id, annotation in annotations.items():
        for obj in annotation['objects']:
            category = obj['category']
            if category in class_counts:
                class_counts[category] += 1
            else:
                class_counts[category] = 1
                
            # Calculate the bounding box area
            bbox = obj['bbox']
            width = bbox['xmax'] - bbox['xmin']
            height = bbox['ymax'] - bbox['ymin']
            area = width * height
            bounding_box_areas.append(area)

    # Filter classes with more than 10 instances
    filtered_class_counts = {label: count for label, count in class_counts.items() if count > 10}

    # Convert to a pandas DataFrame for easy plotting
    df = pd.DataFrame(list(filtered_class_counts.items()), columns=['Class', 'Count'])

    # Sort the DataFrame by count for better visualization
    df = df.sort_values(by='Count', ascending=False)

    # Plot the histogram of class counts
    plt.figure(figsize=(12, 8))
    plt.bar(df['Class'], df['Count'])
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Number of Instances in Each Class (Classes with > 10 Instances)')
    plt.xticks(rotation=90)  # Rotate class labels for better readability
    plt.show()

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

else:
    print("Annotations data is empty or not found.")