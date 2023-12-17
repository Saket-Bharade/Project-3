import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Function to visualize intermediate results for one image
def visualize_intermediate_results(image_path):
    # Read the original image
    original_image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
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
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Plot the results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(142), plt.imshow(edges, cmap='gray'), plt.title('Canny Edge Detection')
    plt.subplot(143), plt.imshow(mask, cmap='gray'), plt.title('Contours Mask')
    plt.subplot(144), plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)), plt.title('Masked Image')
    
    plt.show()

# Specify your dataset path
dataset_path = "C:\\Users\\saket\\Documents\\GitHub\\Project-3\\Project 3 Data\\data"

# Specify the image path
image_path = os.path.join(dataset_path, r"C:\Users\saket\Documents\GitHub\Project-3\Project 3 Data\motherboard_image.JPEG")

# Visualize intermediate results for one image
visualize_intermediate_results(image_path)





