import json
import os
import cv2

# Define the class name to ID mapping
class_mapping = {
    'pl100': 0,
    'pl60': 1,
    'p11': 2,
    'pl40': 3,
    'i2r': 4,
    'il60': 5,
    'pl5': 6,
    'pl30': 7,
    'pn': 8,
    'pne': 9,
    'i2': 10,
    'pl80': 11,
    'p26': 12,
    'i5': 13,
    'p5': 14,
    'pl50': 15,
    'i4': 16,
    'w57': 17,
    'p10': 18
}

def create_yolo_annotations(image_folders, annotation_json, yolo_output_dir):
    # Print paths to debug
    print(f"Annotation JSON path: {annotation_json}")
    print(f"YOLO annotations output directory: {yolo_output_dir}")

    # Check if annotation file exists
    if not os.path.exists(annotation_json):
        print(f"Error: Annotation file does not exist at {annotation_json}")
        return
    
    print("Loading annotations...")
    # Load the annotation JSON file
    with open(annotation_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Ensure the YOLO output directory exists
    if not os.path.exists(yolo_output_dir):
        print(f"Creating YOLO output directory at {yolo_output_dir}")
        os.makedirs(yolo_output_dir)

    for folder in image_folders:
        if not os.path.exists(folder):
            print(f"Error: Image folder does not exist at {folder}")
            continue
        
        print(f"Processing folder: {folder}")
        for img_filename in os.listdir(folder):
            if img_filename.endswith('.jpg'):
                print(f"Processing image: {img_filename}")
                # Load image to get dimensions
                img_path = os.path.join(folder, img_filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Failed to load image at {img_path}")
                    continue
                height, width, _ = img.shape
                print(height)
                
                # Get annotations for this image
                img_id = os.path.splitext(img_filename)[0]
                img_annotations = annotations['imgs'].get(img_id, {}).get('objects', [])
                yolo_annotations = []

                for ann in img_annotations:
                    # Convert COCO bbox to YOLO format
                    bbox = ann["bbox"]
                    x_center = (bbox["xmin"] + (bbox["xmax"] - bbox["xmin"]) / 2) / width
                    y_center = (bbox["ymin"] + (bbox["ymax"] - bbox["ymin"]) / 2) / height
                    bbox_width = (bbox["xmax"] - bbox["xmin"]) / width
                    bbox_height = (bbox["ymax"] - bbox["ymin"]) / height

                    # Convert category name to ID using the mapping
                    category_name = ann['category']
                    if category_name in class_mapping:
                        category_id = class_mapping[category_name]
                    else:
                        print(f"Warning: Unknown category {category_name} in image {img_filename}")
                        continue  # Skip unknown categories

                    yolo_annotation = f"{category_id} {x_center} {y_center} {bbox_width} {bbox_height}"
                    yolo_annotations.append(yolo_annotation)

                # Save YOLO annotations to file
                yolo_annotation_file = os.path.join(yolo_output_dir, os.path.splitext(img_filename)[0] + '.txt')
                with open(yolo_annotation_file, 'w') as f:
                    for yolo_ann in yolo_annotations:
                        f.write(yolo_ann + '\n')


# Define paths to image folders
image_folders = [
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train/images',
]

# Define paths to annotation JSON and YOLO output directory
annotation_json = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train_split.json'
yolo_output_dir = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train/labels'

# Create YOLO annotations
create_yolo_annotations(image_folders, annotation_json, yolo_output_dir)
