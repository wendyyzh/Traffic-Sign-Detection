import glob
from roboflow import Roboflow

# Initialize Roboflow client
rf = Roboflow(api_key="z0cr2Pq9nzZdnfIVptSz")


# Directory paths for images
dir_names = [
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/marks',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/train',
    'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/test'
]
file_extension_type = ".jpg"

# Annotation file path and format (e.g., .coco.json)
annotation_filename = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/coco_all.json'

# Get the upload project from Roboflow workspace
project = rf.workspace().project("traffic_sign_recognition-bq9mi")

# Upload images
for dir_name in dir_names:
    image_glob = glob.glob(f'{dir_name}/*{file_extension_type}')
    for image_path in image_glob:
        print(project.single_upload(
            image_path=image_path,
            annotation_path=annotation_filename,
            # optional parameters:
            # annotation_labelmap=labelmap_path,
            # split='train',
            # num_retry_uploads=0,
            # batch_name='batch_name',
            # tag_names=['tag1', 'tag2'],
            # is_prediction=False,
        ))
