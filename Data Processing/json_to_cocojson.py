import json
import os
import cv2

def create_coco_json(image_folders, annotation_json, output_json):
    # Print paths to debug
    print(f"Annotation JSON path: {annotation_json}")
    print(f"Output JSON path: {output_json}")

    # Check if annotation file exists
    if not os.path.exists(annotation_json):
        print(f"Error: Annotation file does not exist at {annotation_json}")
        return
    
    print("Loading annotations...")
    # Load the annotation JSON file
    with open(annotation_json, 'r') as f:
        annotations = json.load(f)
    
    print("Initializing COCO JSON structure...")
    # Initialize COCO JSON structure
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Assuming categories are predefined
    categories = [{"id": 1, "name": "category1"}, {"id": 2, "name": "category2"}]  # Add your categories
    coco_json["categories"] = categories

    annotation_id = 1

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
                
                # Create image entry
                img_id = int(os.path.splitext(img_filename)[0])
                image_entry = {
                    "id": img_id,
                    "width": width,
                    "height": height,
                    "file_name": img_filename
                }
                coco_json["images"].append(image_entry)
                
                # Get annotations for this image
                img_annotations = annotations.get(str(img_id), [])
                
                for ann in img_annotations:
                    annotation_entry = {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": ann["category_id"],
                        "segmentation": ann.get("segmentation", []),
                        "area": ann["area"],
                        "bbox": ann["bbox"],
                        "iscrowd": ann.get("iscrowd", 0)
                    }
                    coco_json["annotations"].append(annotation_entry)
                    annotation_id += 1

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_json)
    if not os.path.exists(output_dir):
        print(f"Creating output directory at {output_dir}")
        os.makedirs(output_dir)

    print("Saving COCO JSON to file...")
    # Save COCO JSON to file
    with open(output_json, 'w') as f:
        json.dump(coco_json, f, indent=4)

    print("COCO JSON creation completed.")

# Define paths to image folders
image_folders = [
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/marks',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test'
]

# Define paths to annotation JSON and output JSON
annotation_json = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/annotations_all.json'
output_json = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/coco_all.json'

# Create COCO JSON
create_coco_json(image_folders, annotation_json, output_json)
