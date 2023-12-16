import cv2
import os
import numpy as np
from pathlib import Path

# Step 1: Object Masking

def object_masking(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, file_name)

            # Read the image
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise and improve edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an empty mask
            mask = np.zeros_like(edges)

            # Draw contours on the mask
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

            # Bitwise AND to extract the PCB from the background
            motherboard = cv2.bitwise_and(image, image, mask=mask)

            # Save the masked image to the output directory
            masked_image_path = os.path.join(output_dir, file_name)
            cv2.imwrite(masked_image_path, motherboard)

# Specify your dataset path
dataset_path = "C:\\Users\\saket\\Documents\\GitHub\\Project-3\\Project 3 Data\\data"

# Apply object masking to the training set images
train_images_dir = os.path.join(dataset_path, "train", "images")
masked_train_dir = os.path.join(dataset_path, "masked_train")

object_masking(train_images_dir, masked_train_dir)