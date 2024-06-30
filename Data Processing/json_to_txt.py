import json
import os
import cv2

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
                
                # Get annotations for this image
                img_id = int(os.path.splitext(img_filename)[0])
                img_annotations = annotations['imgs'].get(str(img_id), {}).get('objects', [])
                yolo_annotations = []


                for ann in img_annotations:

                    # Convert COCO bbox to YOLO format
                    bbox = ann["bbox"]
                    x_center = (bbox["xmin"] + (bbox["xmax"] - bbox["xmin"]) / 2) / width
                    y_center = (bbox["ymin"] + (bbox["ymax"] - bbox["ymin"]) / 2) / height
                    bbox_width = (bbox["xmax"] - bbox["xmin"]) / width
                    bbox_height = (bbox["ymax"] - bbox["ymin"]) / height

                    # The category should be converted to an ID; if mapping exists, apply here
                    category_id = ann['category']  # This might need conversion if using string names

                    yolo_annotation = f"{category_id} {x_center} {y_center} {bbox_width} {bbox_height}"
                    yolo_annotations.append(yolo_annotation)


                # Save YOLO annotations to file
                yolo_annotation_file = os.path.join(yolo_output_dir, os.path.splitext(img_filename)[0] + '.txt')
                with open(yolo_annotation_file, 'w') as f:
                    for yolo_ann in yolo_annotations:
                        f.write(yolo_ann + '\n')

# Define paths to image folders
image_folders = [
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/marks',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test'
]

# Define paths to annotation JSON and YOLO output directory
annotation_json = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/annotations_all.json'
yolo_output_dir = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/yolo_annotations'

# Create YOLO annotations
create_yolo_annotations(image_folders, annotation_json, yolo_output_dir)