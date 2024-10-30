import cv2
import numpy as np
import os
import shutil

# Function to apply rotation to an image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(f"{output_folder}/img_{angle}.png", rotated)
    return rotated

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    return thresh

# Function to split the image into lines using contour detection
def split_into_lines(thresh_img):
    # Find contours of the text regions
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours top-to-bottom to maintain reading order
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[1])
    
    lines = []
    for box in sorted_boxes:
        x, y, w, h = box
        lines.append((y, y + h, x, x + w))
    
    return lines

# Function to evaluate the quality of line splitting
def evaluate_line_split(lines):
    # Evaluation can be based on number of lines and their bounding box sizes
    num_lines = len(lines)
    total_line_height = sum([line[1] - line[0] for line in lines])
    print(num_lines, total_line_height, sep=" ")
    
    # Return a score (higher is better), you can customize this metric
    return num_lines * total_line_height

# Function to find the best tilt for line splitting
def find_best_tilt(image, max_angle=0, step=1):
    best_score = -1
    best_lines = None
    best_angle = 0

    for angle in range(-max_angle, max_angle + 1, step):
        # Rotate the image
        rotated_img = rotate_image(image, angle)
        
        # Try splitting into lines
        lines = split_into_lines(rotated_img)
        
        # Evaluate the splitting
        score = evaluate_line_split(lines)
        # print(angle, score, sep=" ")
        
        # Keep track of the best angle and lines
        if score > best_score:
            best_score = score
            best_lines = lines
            best_angle = angle
    
    # Rotate the image using the best angle
    best_rotated_image = rotate_image(image, best_angle)
    
    return best_rotated_image, best_lines, best_angle

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
    # Preprocess the image
    thresh_img = preprocess_image(image_path)
    
    # Find the best tilt and the corresponding lines
    best_img, best_lines, best_angle = find_best_tilt(thresh_img)
    print(f"Best tilt angle: {best_angle} degrees")
    
    letter_count = 0
    
    for idx, line in enumerate(best_lines):
        # Split each line into letters
        letters, line_img = split_line_into_letters(best_img, line)
        
        for jdx, (x, y, w, h) in enumerate(letters):
            # Extract the letter image
            letter_img = line_img[y:y + h, x:x + w]
            
            # Save the letter image
            letter_filename = f"{output_folder}/letter_{idx}_{jdx}.png"
            cv2.imwrite(letter_filename, letter_img)
            
            letter_count += 1
    
    print(f"Extracted {letter_count} letters.")

# Example usage
# image_path = 'path_to_image.png'  # Path to your image containing text
output_folder = 'output_letters_2'  # Folder where the extracted letters will be saved

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Call the function to extract letters
# extract_letters("k1.jpg", output_folder)
# extract_letters("a1.jpg", output_folder)
extract_letters("l11.jpg", output_folder)
