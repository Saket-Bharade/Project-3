from PIL import Image
import os
from IPython import display

display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model_path = r"C:\Users\saket\Downloads\best.pt"
# Load a pretrained YOLOv8n model
model = YOLO(model_path)

# Define path to the image file
source = r"C:\Users\saket\Documents\GitHub\Project-3\Project 3 Data\data\evaluation"

# Run inference on the source
results = model(source)

for i, r in enumerate(results):
    # Save the images back to the same folder with a new name
    save_path = os.path.join(source, f"output_image_{i}.jpg")
    
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.save(save_path)
    print(f"Image saved to: {save_path}")
