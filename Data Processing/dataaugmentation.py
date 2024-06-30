import json
import os
import cv2
import numpy as np
import random

# Paths to annotation JSON file and multiple image directories
annotation_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/annotations_all.json'
image_base_paths = [
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/other',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/marks'
]
output_dir = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/augmented_images'

# Load the JSON annotation file
with open(annotation_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Verify the structure of the loaded data
print(f"Data type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys in the data: {data.keys()}")
else:
    print("Data is not a dictionary.")

# Get annotations
annotations = data.get('imgs', None)

if annotations is None:
    print("Error: 'imgs' key not found in the data.")
else:
    # Initialize class counts
    augmentation_class_counts = {}
    for img_id, annotation in annotations.items():
        for obj in annotation['objects']:
            category = obj['category']
            if category in augmentation_class_counts:
                augmentation_class_counts[category] += 1
            else:
                augmentation_class_counts[category] = 1

    augmentation_class_counts = {k: v for k, v in augmentation_class_counts.items() if 500 <= v < 1000}

    os.makedirs(output_dir, exist_ok=True)

    def find_image(img_filename):
        for base_path in image_base_paths:
            img_path = os.path.join(base_path, img_filename)
            if os.path.exists(img_path):
                return img_path
        return None

    def augment_image(img, bbox):
        augments = []

        # Random Flipping
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            for box in bbox:
                box['xmin'], box['xmax'] = 1 - box['xmax'], 1 - box['xmin']
            augments.append("flip")

        # Random Scaling
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            h, w, _ = img.shape
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            for box in bbox:
                box['xmin'], box['xmax'] = box['xmin'] * scale, box['xmax'] * scale
                box['ymin'], box['ymax'] = box['ymin'] * scale, box['ymax'] * scale
            augments.append("scale")

        # Random Translation
        if random.random() > 0.5:
            tx = random.randint(-10, 10)
            ty = random.randint(-10, 10)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h))
            for box in bbox:
                box['xmin'] += tx / w
                box['xmax'] += tx / w
                box['ymin'] += ty / h
                box['ymax'] += ty / h
            augments.append("translate")

        # Random Rotation
        if random.random() > 0.5:
            angle = random.randint(-10, 10)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
            for box in bbox:
                x = (box['xmin'] + box['xmax']) / 2
                y = (box['ymin'] + box['ymax']) / 2
                box['xmin'], box['ymin'] = rotate_point((x, y), (box['xmin'], box['ymin']), angle)
                box['xmax'], box['ymax'] = rotate_point((x, y), (box['xmax'], box['ymax']), angle)
            augments.append("rotate")

        # Random Brightness/Contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-10, 10)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            augments.append("brightness_contrast")

        # Random Saturation/Hue
        if random.random() > 0.5:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = np.array(img, dtype=np.float64)
            img[..., 1] = img[..., 1] * random.uniform(0.8, 1.2)
            img[..., 2] = img[..., 2] * random.uniform(0.8, 1.2)
            img = np.clip(img, 0, 255)
            img = np.array(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            augments.append("saturation_hue")

        return img, bbox, augments

    def rotate_point(center, point, angle):
        angle = np.deg2rad(angle)
        temp_point = point[0] - center[0], point[1] - center[1]
        temp_point = (temp_point[0] * np.cos(angle) - temp_point[1] * np.sin(angle),
                      temp_point[0] * np.sin(angle) + temp_point[1] * np.cos(angle))
        temp_point = temp_point[0] + center[0], temp_point[1] + center[1]
        return temp_point

    augmented_annotations = {}

    for img_id, annotation in annotations.items():
        img_filename = annotation['path']
        img_path = find_image(img_filename)
        if img_path is None:
            print(f"Failed to find image: {img_filename}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Collect bounding boxes
        bbox = [{'xmin': obj['bbox']['xmin'], 'ymin': obj['bbox']['ymin'],
                 'xmax': obj['bbox']['xmax'], 'ymax': obj['ymax'],
                 'category': obj['category']} for obj in annotation['objects']]

        for i in range(2):  # Augment each image twice
            aug_img, aug_bbox, augments = augment_image(img.copy(), bbox.copy())
            aug_img_id = f"{img_id}_aug_{i}"
            aug_img_path = os.path.join(output_dir, f"{aug_img_id}.jpg")
            cv2.imwrite(aug_img_path, aug_img)

            augmented_annotations[aug_img_id] = {
                'path': aug_img_path,
                'id': aug_img_id,
                'objects': [{'bbox': {'xmin': box['xmin'], 'ymin': box['ymin'], 'xmax': box['xmax'], 'ymax': box['ymax']},
                             'category': box['category']} for box in aug_bbox]
            }

    # Save augmented annotations
    augmented_file_path = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/augmented_annotations.json'
    with open(augmented_file_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_annotations, f, ensure_ascii=False, indent=4)