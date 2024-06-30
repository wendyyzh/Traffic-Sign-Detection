import json
import os
import cv2

def create_coco_json(jpg_folder, annotation_json, output_json):
    # Load the annotation JSON file
    with open(annotation_json, 'r') as f:
        annotations = json.load(f)
    
    # Initialize COCO JSON structure
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Assuming categories are predefined
    categories = [{"id": 1, "name": "category1"}, {"id": 2, "name": "category2"}] # Add your categories
    coco_json["categories"] = categories

    annotation_id = 1

    for img_filename in os.listdir(jpg_folder):
        if img_filename.endswith('.jpg'):
            # Load image to get dimensions
            img_path = os.path.join(jpg_folder, img_filename)
            img = cv2.imread(img_path)
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

    # Save COCO JSON to file
    with open(output_json, 'w') as f:
        json.dump(coco_json, f, indent=4)

# Define paths
jpg_folder = 'path_to_your_jpg_folder'
annotation_json = 'path_to_your_annotation_json_file'
output_json = 'path_to_output_coco_json_file'

# Create COCO JSON
create_coco_json(jpg_folder, annotation_json, output_json)