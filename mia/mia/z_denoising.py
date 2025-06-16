#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 01:32:22 2025

@author: surajit
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from skimage.morphology import white_tophat, disk
#pip install filterpy
from scipy.interpolate import griddata
from filterpy.kalman import KalmanFilter

def show_img(img_in, img_out):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1), plt.imshow(img_in, cmap='gray'), plt.title("Original Image"), plt.axis("off")
    plt.subplot(1,2,2), plt.imshow(img_out, cmap='gray'), plt.title("Frequency Filtered"), plt.axis("off")
    plt.show()
    




import cv2
import numpy as np
from scipy.interpolate import griddata

def detect_circles(image):
    """Detects circular objects using Hough Transform."""
    if image is None:
        print("Error: Input image is empty.")
        return np.array([])  # Return an empty array

    # Ensure the image is grayscale
    if len(image.shape) != 2:
        print("Error: Input image must be grayscale.")
        return np.array([])  # Return an empty array

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=30)
    return np.array([]) if circles is None else circles[0]

def interpolate_background(image, mask):
    """Interpolates the background using surrounding pixels."""
    # Get the coordinates of the masked region
    points = np.column_stack(np.where(mask == 0))  # Background pixels
    values = image[mask == 0]  # Pixel values of the background

    # Get the coordinates of the masked region (circles)
    missing_points = np.column_stack(np.where(mask == 1))  # Circle pixels

    # Interpolate the missing pixels
    if len(points) > 0 and len(missing_points) > 0:
        interpolated_values = griddata(points, values, missing_points, method='cubic')
        image[mask == 1] = interpolated_values  # Fill the circles with interpolated values

    return image

def Hough_Circle_Transform(image_path, status_label):
    """
    Detects circular blobs using Hough Transform, removes them, and fills with interpolated background.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        status_label.config(text=f"Error: Could not load image from {image_path}")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles
    circles = detect_circles(gray)
    if circles.size == 0:  # Check if no circles are detected
        status_label.config(text="No circles detected.")
        return image, image

    # Create a mask for the circles
    mask = np.zeros_like(gray)
    for circle in circles:
        x, y, r = circle
        x, y, r = int(x), int(y), int(r)  # Convert to integers
        cv2.circle(mask, (x, y), r, 1, -1)  # Fill the circle with 1

    # Invert the mask (1 for circles, 0 for background)
    mask = mask.astype(np.uint8)

    # Interpolate the background for each channel (BGR)
    output_image = image.copy()
    for channel in range(3):  # Process each channel (B, G, R)
        output_image[:, :, channel] = interpolate_background(image[:, :, channel], mask)

    # Save the processed image
    output_file = f"processed_{os.path.basename(image_path)}"
    cv2.imwrite(output_file, output_image)

    # Update status label
    status_label.config(text=f"Output saved as {output_file}")

    # Display results
    show_img(image, output_image)

    return image, output_image





def hough(image_path, status_label):
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    prefix="x"
    # Create output file name
    output_file = f"{prefix}_{file_name}{file_extension}"
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply median blur to reduce noise for better circle detection
    blurred = cv2.medianBlur(image, 5)
    
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=20)
    
    # Create a mask for inpainting
    mask = np.zeros_like(image, dtype=np.uint8)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Round circle coordinates
        for i in circles[0, :]:
            # Draw filled circles on the mask
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)
    
    # Inpaint the detected circles using Telea's method
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    # Save the results
    cv2.imwrite(output_file, inpainted_image)
    status_label.config(text=f"Output saved as {prefix}_{file_name}{file_extension}")
    show_img(image, inpainted_image)
    
    return image, inpainted_image





def Connected_Component_Analysis(image_path, status_label):
 
    from skimage import measure

    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    prefix = "c"
    # Create output file name for the processed image
    output_file = f"{prefix}_{file_name}{file_extension}"

    # Load the grayscale image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        status_label.config(text="Error: Image not found or could not be read.")
        return None

    # Apply thresholding to get a binary image; here, THRESH_BINARY_INV is used
    # so that the foreground (objects) becomes white (255) and background black (0)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Convert binary image to boolean for scikit-image processing
    binary_bool = binary > 0

    # Label connected regions using scikit-image (connectivity=2 gives 8-connected components)
    label_image = measure.label(binary_bool, connectivity=2)

    # Compute region properties using scikit-image's regionprops
    regions = measure.regionprops(label_image)

    # Create a list to store region statistics
    region_data = []
    for region in regions:
        area = region.area
        centroid = region.centroid  # (row, col)
        bbox = region.bbox          # (min_row, min_col, max_row, max_col)
        eccentricity = region.eccentricity
        perimeter = region.perimeter
        # Compute isoperimetric ratio: (4 * π * Area) / (Perimeter^2)
        iso_ratio = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan

        region_data.append({
            "Label": region.label,
            "Area": area,
            "Centroid": centroid,
            "Bounding Box": bbox,
            "Eccentricity": eccentricity,
            "Isoperimetric Ratio": iso_ratio
        })

    # (Optional) Print region data to console for debugging
    for data in region_data:
        print(data)

    # Save the processed binary image using OpenCV
    cv2.imwrite(output_file, binary)

    # Update the status label with the output file name
    status_label.config(text=f"Output saved as {output_file}")

    # Display the original grayscale image and the processed binary image
    # (Assuming that show_img is defined elsewhere in your project)
    show_img(image, binary)

    return image, binary, region_data














'''
def Connected_Component_Analysis(image_path, status_label):
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    prefix="c"
    # Create output file name
    output_file = f"{prefix}_{file_name}{file_extension}"

    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply Connected Component Analysis (CCA)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create output image
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Assign random colors to each component
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)

    colored_labels = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

    for label in range(1, num_labels):  # Skipping background
        colored_labels[labels == label, :] = colors[label]
    

    # Extract component statistics
    components_info = []
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        centroid_x, centroid_y = centroids[i]
        components_info.append({
            "Label": i,
            "Area": area,
            "Centroid": (centroid_x, centroid_y),
            "Bounding Box": (x, y, w, h)
        })
    
    # Print first few component details
    for comp in components_info[:5]:
        print(comp)
    
    # Save the processed image
    cv2.imwrite(output_file, colored_labels)
    status_label.config(text=f"Output saved as {prefix}_{file_name}{file_extension}")
    show_img(image, colored_labels)
    return image, colored_labels





def Connected_Component_Analysis(image_path, status_label):
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    prefix="c"
    # Create output file name
    output_file = f"{prefix}_{file_name}{file_extension}"

    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Perform Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter out small regions (noise)
    min_area = 50
    filtered_labels = np.zeros_like(labels)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_labels[labels == i] = i
    
    # Visualize the results
    output = cv2.merge([image, image, image])
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            color = np.random.randint(0, 255, 3)
            output[labels == i] = color
    
    # Save the processed image
    cv2.imwrite(output_file, output)
    status_label.config(text=f"Output saved as {prefix}_{file_name}{file_extension}")
    show_img(image, output)
    return image, output





def Connected_Component_Analysis(image_path, status_label):
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    prefix="c"
    # Create output file name
    output_file = f"{prefix}_{file_name}{file_extension}"

    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    
    # Create an output image
    output = np.zeros_like(image)
    
    # Filter out small components (e.g., noise) based on area
    min_area = 5  # Adjust this threshold as needed
    
    for i in range(1, num_labels):  # Start from 1 to exclude the background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255  # Keep only large connected components
    
    # Invert the result to match the original image format
    output = cv2.bitwise_not(output)
    
    # Save the processed image
    cv2.imwrite(output_file, output)
    status_label.config(text=f"Output saved as {prefix}_{file_name}{file_extension}")
    show_img(image, output)
    return image, output
'''


  
    

def Wavelet_Transform():
    pass


def Denoising_Autoencoders():
    pass

def Unet_GAN():
    pass

def Gausian_Median():
    pass

def Rolling_Ball(image_path, status_label):
    from skimage.morphology import white_tophat, ball
    
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    prefix="R"
    # Create output file name
    output_file = f"{prefix}_{file_name}{file_extension}"   
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 50  # Adjust as needed
    footprint = disk(radius)  # Use disk instead of ball
    background = white_tophat(image, footprint=footprint)
    
    # Save the filtered image
    cv2.imwrite(output_file, image-background)
    
    # Update status label
    status_label.config(text=f"Output saved as {prefix}_{file_name}{file_extension}")
    
    # Display the original and filtered images (assuming show_img is defined elsewhere)
    show_img(image, image-background) 
   
    return image, image - background












def Bilat(image_path, status_label):
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    prefix = "BF"  # Prefix for Bilateral Filtering
    # Create output file name
    output_file = f"{prefix}_{file_name}{file_extension}"
    
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply bilateral filtering
    d = 9  # Diameter of each pixel neighborhood
    sigma_color = 75  # Filter sigma in the color space
    sigma_space = 75  # Filter sigma in the coordinate space
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    # Save the filtered image
    cv2.imwrite(output_file, filtered_image)
    
    # Update status label
    status_label.config(text=f"Output saved as {prefix}_{file_name}{file_extension}")
    
    # Display the original and filtered images (assuming show_img is defined elsewhere)
    show_img(image, filtered_image)
    
    return image, filtered_image




def NLMn(image_path, status_label):
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    prefix = "NLM"  # Prefix for Non-Local Means Filtering
    # Create output file name
    output_file = f"{prefix}_{file_name}{file_extension}"
    
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Non-Local Means Filtering
    h = 10  # Filter strength (higher value removes more noise but may blur details)
    template_window_size = 7  # Size of the patch used for comparison
    search_window_size = 21  # Size of the region to search for similar patches
    filtered_image = cv2.fastNlMeansDenoising(image, h=h, templateWindowSize=template_window_size, searchWindowSize=search_window_size)
    
    # Save the filtered image
    cv2.imwrite(output_file, filtered_image)
    
    # Update status label
    status_label.config(text=f"Output saved as {prefix}_{file_name}{file_extension}")
    
    # Display the original and filtered images (assuming show_img is defined elsewhere)
    show_img(image, filtered_image)
    
    return image, filtered_image
    
    
    
    
    
    
    
    
    
    
    
    
    
    

