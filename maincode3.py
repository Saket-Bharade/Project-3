import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Function to visualize intermediate results for one image
def visualize_intermediate_results(image_path, brightness_factor=1):
    # Read the original image
    original_image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color of the PCB, considering glare
    lower_bound = np.array([20, 50, 50])
    upper_bound = np.array([40, 255, 255])

    # Create a mask using inRange
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Dilate the mask to cover glare areas
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Use morphological operations to further improve the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Bitwise AND to extract the PCB from the background
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Brighten the masked image
    brightened_image = cv2.addWeighted(masked_image, brightness_factor, np.zeros_like(masked_image), 0, 0)

    # Plot the results
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(142), plt.imshow(mask, cmap='gray'), plt.title('Color-based Mask')
    plt.subplot(143), plt.imshow(cv2.cvtColor(brightened_image, cv2.COLOR_BGR2RGB)), plt.title('Brightened Masked Image')
    
    plt.show()

# Specify your dataset path
dataset_path = "C:\\Users\\saket\\Documents\\GitHub\\Project-3\\Project 3 Data\\data"

# Specify the image path
image_path = os.path.join(dataset_path, r"C:\Users\saket\Documents\GitHub\Project-3\Project 3 Data\motherboard_image.JPEG")

# Visualize intermediate results for one image with default parameters
visualize_intermediate_results(image_path)




