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
    return rotated

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    return thresh

# # Function to split the image into lines using contour detection
# def split_into_lines(thresh_img):
#     # Find contours of the text regions
#     contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Sort contours top-to-bottom to maintain reading order
#     bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
#     sorted_boxes = sorted(bounding_boxes, key=lambda box: box[1])
    
#     lines = []
#     for box in sorted_boxes:
#         x, y, w, h = box
#         lines.append((y, y + h, x, x + w))
    
#     return lines

# Function to split the image into lines
def split_into_lines(thresh_img):
    # Sum up pixel values vertically to find text lines
    horizontal_sum = np.sum(thresh_img, axis=1)
    
    # Find boundaries of lines
    lines = []
    line_start = None
    threshold = 128  # Adjust threshold to control sensitivity

    for i, value in enumerate(horizontal_sum):
        if value > threshold and line_start is None:
            line_start = i
        elif value < threshold and line_start is not None:
            lines.append((line_start, i))
            line_start = None
    print(lines)
    return lines

# Function to evaluate the quality of line splitting
def evaluate_line_split(lines):
    # Evaluation can be based on number of lines and their bounding box sizes
    num_lines = len(lines)
    total_line_height = sum([line[1] - line[0] for line in lines])
    print(num_lines, total_line_height, sep=" ", end=" ")
    
    # Return a score (higher is better), you can customize this metric
    return num_lines * total_line_height

# Function to find the best tilt for line splitting
def find_best_tilt(image, max_angle=5, step=1):
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
        print(score, sep=" ")
        
        # Keep track of the best angle and lines
        if score > best_score:
            best_score = score
            best_lines = lines
            best_angle = angle
    
    # Rotate the image using the best angle
    best_rotated_image = rotate_image(image, best_angle)
    
    return best_rotated_image, best_lines, best_angle

# Function to split a line into words by detecting spaces between words
def split_line_into_words(thresh_img, line, space_threshold=15):
    # Crop the line from the image
    line_img = thresh_img[line[0]:line[1], :]
    
    # Find contours for letters within the line
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left-to-right to maintain reading order
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda box: box[0])
    
    # Detect gaps between bounding boxes to split into words
    words = []
    current_word = []
    prev_x_end = None
    
    for box in sorted_boxes:
        x, y, w, h = box
        if prev_x_end is not None and (x - prev_x_end) > space_threshold:
            # If gap between words exceeds threshold, treat it as a word boundary
            if current_word:
                words.append(current_word)
            current_word = []
        
        current_word.append(box)
        prev_x_end = x + w
    
    if current_word:
        words.append(current_word)
    
    return words, line_img

# Function to extract and save individual words
def extract_words(image_path, output_folder):
    # Preprocess the image
    thresh_img = preprocess_image(image_path)
    
    # Find the best tilt and the corresponding lines
    best_img, best_lines, best_angle = find_best_tilt(thresh_img)
    print(f"Best tilt angle: {best_angle} degrees")
    
    word_count = 0
    
    for idx, line in enumerate(best_lines):
        # Split each line into words
        words, line_img = split_line_into_words(best_img, line)
        
        for jdx, word_boxes in enumerate(words):
            # Merge the bounding boxes of the word to crop the word from the line
            min_x = min([box[0] for box in word_boxes])
            max_x = max([box[0] + box[2] for box in word_boxes])
            min_y = min([box[1] for box in word_boxes])
            max_y = max([box[1] + box[3] for box in word_boxes])
            
            word_img = line_img[min_y:max_y, min_x:max_x]
            
            # Save the word image
            word_filename = f"{output_folder}/word_{idx}_{jdx}.png"
            cv2.imwrite(word_filename, word_img)
            
            word_count += 1
    
    print(f"Extracted {word_count} words.")

# Example usage
# image_path = 'path_to_image.png'  # Path to your image containing text
output_folder = 'output_letters'  # Folder where the extracted letters will be saved

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Call the function to extract letters
extract_words("k1.jpg", output_folder)
# extract_words("a1.jpg", output_folder)
# extract_words("l1.jpg", output_folder)
