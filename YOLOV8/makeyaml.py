import os
import yaml

# Define the dataset path and the folder structure
dataset_path = "C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021"
train_images_path = os.path.join(dataset_path, 'train', 'images')
val_images_path = os.path.join(dataset_path, 'validation', 'images')

# Define the content of the data.yaml file
data = {
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
    'nc': 19,
    'train': train_images_path,
    'val': val_images_path
}

# Path to the data.yaml file
data_yaml_path = os.path.join(dataset_path, 'data.yaml')

# Create the data.yaml file
with open(data_yaml_path, 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

print(f"data.yaml file created successfully at {data_yaml_path}")
