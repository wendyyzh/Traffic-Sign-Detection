import os
import yaml

# Define the dataset path and the folder structure
dataset_path = "C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021"
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')
train_images_path = os.path.join(images_path, 'train_split')
val_images_path = os.path.join(images_path, 'val')
train_labels_path = os.path.join(labels_path, 'train_txt')
val_labels_path = os.path.join(labels_path, 'val_txt')

# Create the directory structure
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# Define the content of the data.yaml file
data = {
    'train': train_images_path,
    'val': val_images_path,
    'nc': 19,
    'names': [
        "pl100",
        "pl60",
        "p11",
        "pl40",
        "i2r",
        "il60",
        "pl5",
        "pl30",
        "pn",
        "pne",
        "i2",
        "pl80",
        "p26",
        "i5",
        "p5",
        "pl50",
        "i4",
        "w57",
        "p10"
    ],
}

# Path to the data.yaml file
data_yaml_path = os.path.join(dataset_path, 'data.yaml')

# Create the data.yaml file
with open(data_yaml_path, 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

print(f"data.yaml file created successfully at {data_yaml_path}")
print(f"Directory structure created under {dataset_path}")