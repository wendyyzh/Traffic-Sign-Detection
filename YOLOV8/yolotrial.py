from ultralytics import YOLO
from IPython.display import Image, display
import os

# Ensure the dataset location is correctly set
dataset_location = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021'

# Initialize the model
print("Initializing the model...")
model = YOLO('yolov8m.pt')  # or specify the path to your model

# Train the model
print("Starting training...")
model.train(data=os.path.join(dataset_location, 'data.yaml'), epochs=20, imgsz=640)
print("Training completed.")

# Display the confusion matrix and results images
confusion_matrix_path = "runs/detect/train/confusion_matrix.png"
results_path = "runs/detect/train/results.png"

print("Displaying training results...")

if os.path.exists(confusion_matrix_path):
    display(Image(filename=confusion_matrix_path))
else:
    print(f"Confusion matrix not found at {confusion_matrix_path}")

if os.path.exists(results_path):
    display(Image(filename=results_path, width=600))
else:
    print(f"Results image not found at {results_path}")

# Validate the model
print("Validating the model...")
model.val(data=os.path.join(dataset_location, 'data.yaml'))
print("Validation completed.")

# Predict with the model
source = 'C:/Users/wezha/OneDrive/Desktop/tt100k_2021/tt100k_2021/images/val'  # Path to the images for prediction
print("Running predictions...")
results = model.predict(source=source)
print("Predictions completed.")

# Display prediction results
print("Displaying prediction results...")
for img_path in results.files:
    display(Image(filename=img_path))

print("Script completed successfully.")
