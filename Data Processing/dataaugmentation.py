import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
import cv2


# Path to the JSON file containing filtered annotations for augmentation
augmentation_file_path = '/Users/jessica_1/Documents/tt100k_2021/annotations_for_augmentation.json'

# Load the JSON file
with open(augmentation_file_path, 'r', encoding='utf-8') as f:
    augmentation_data = json.load(f)


tf.keras.utils.get_file('flower_photos', origin=dataset_url, cach_dir = '.', untar = True)