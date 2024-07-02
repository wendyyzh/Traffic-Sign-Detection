import json
import os

# Paths to the directories and annotation files
annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/annotations_all.json'
train_dir = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train'
test_dir = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test'
train_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train.json'
test_annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test.json'

# Load the original annotation file
with open(annotation_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize new dictionaries for train and test annotations
train_annotations = {'types': data['types'], 'imgs': {}}
test_annotations = {'types': data['types'], 'imgs': {}}

# Split the annotations based on the directory of the images
for img_id, annotation in data.get('imgs', {}).items():
    img_filename = annotation['path']
    if img_filename.startswith("train"):
        train_annotations['imgs'][img_id] = annotation
    elif img_filename.startswith("test"):
        test_annotations['imgs'][img_id] = annotation

# Save the train annotations to a new JSON file
with open(train_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(train_annotations, f, ensure_ascii=False, indent=4)

# Save the test annotations to a new JSON file
with open(test_annotation_file_path, 'w', encoding='utf-8') as f:
    json.dump(test_annotations, f, ensure_ascii=False, indent=4)

print(f"Train annotations saved to {train_annotation_file_path}")
print(f"Test annotations saved to {test_annotation_file_path}")