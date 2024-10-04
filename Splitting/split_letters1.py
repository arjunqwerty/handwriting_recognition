import cv2
import numpy as np
import math
import os
import shutil

# Function to deskew the image
def deskew_image(thresh_img):
    angles = [-5, 5, -10, 10, -15, 15]
    desk = []
    for angle in angles:
        # Rotate the image to deskew it
        (h, w) = thresh_img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        deskewed = cv2.warpAffine(thresh_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        desk.append(deskewed) 
    return desk

# Function to preprocess the image and handle margins
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imwrite("blah1.png", thresh)
    
    # Deskew the image
    deskewed = [thresh]
    deskewed.extend(deskew_image(thresh))
    # cv2.imwrite("blah.png", deskewed)
    return deskewed

# Function to split the image into lines using contour detection
def split_into_lines(thresh_img):
    # Find contours of the text regions
    contours, _ = cv2.findContours(thresh_img[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours top-to-bottom to maintain reading order
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[1])
    
    lines = []
    for box in sorted_boxes:
        x, y, w, h = box
        lines.append((y, y + h, x, x + w))
    
    return lines

# Function to split a line into letters using contours
def split_line_into_letters(thresh_img, line):
    # Crop the line from the image
    line_img = thresh_img[line[0]:line[1], line[2]:line[3]]
    
    # Find contours for letters within the line
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left-to-right to maintain reading order
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[0])
    
    return sorted_boxes, line_img

# Function to extract and save individual letters
def extract_letters(image_path, output_folder):
    # minw, minh = 1000, 1000
    # size = []
    # Preprocess the image
    thresh_img = preprocess_image(image_path)
    
    # Split image into lines
    lines = split_into_lines(thresh_img)
    letter_count = 0
    for idx, line in enumerate(lines):
        # Split each line into letters
        letters, line_img = split_line_into_letters(thresh_img[0], line)
        cv2.imwrite(f"{output_folder}/lines_{idx}.png", line_img)
        for jdx, (x, y, w, h) in enumerate(letters):
            # minh = min(minh, h)
            # size.append([h, w])
            # minw = min(minw, w)
            # Extract the letter image
            letter_img = line_img[y:y + h, x:x + w]
            # Save the letter image
            letter_filename = f"{output_folder}/letter_{idx}_{jdx}_{h}_{w}.png"
            cv2.imwrite(letter_filename, letter_img)
            letter_count += 1
    print(f"Extracted {letter_count} letters.")
            
    # print(minh, minw, sep = "  ")
    # print(size)

# Example usage
# image_path = 'path_to_image.png'  # Path to your image containing text
output_folder = 'output_letters'  # Folder where the extracted letters will be saved

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Call the function to extract letters
# extract_letters("k1.jpg", output_folder)
extract_letters("a1.jpg", output_folder)
# extract_letters("l1.jpg", output_folder)
