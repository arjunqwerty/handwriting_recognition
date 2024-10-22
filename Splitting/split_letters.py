import cv2
import numpy as np

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("blah.png", thresh)
    return thresh

# Function to split the image into lines
def split_into_lines(thresh_img):
    # Sum up pixel values vertically to find text lines
    horizontal_sum = np.sum(thresh_img, axis=1)
    # w = thresh_img.shape[1]
    # whitepixel = [np.sum(i==255) for i in thresh_img]
    # for i in whitepixel:
    #     f.write(str(w) + ", " + str(i) + "\n")

    # Find boundaries of lines
    lines = []
    line_start = None
    threshold = 255*5  # Adjust threshold to control sensitivity

    # for i,j in enumerate(whitepixel):
    #     if j > threshold and line_start is None:
    #         line_start = i
    #     elif j < threshold and line_start is not None:
    #         lines.append((line_start, i))
    #         line_start = None

    for i, value in enumerate(horizontal_sum):
        if value > threshold and line_start is None:
            line_start = i
        elif value < threshold and line_start is not None:
            lines.append((line_start, i))
            line_start = None
    # print(lines)
    return lines

# Function to split a line into letters
def split_line_into_letters(thresh_img, line):
    # Crop the line from the image
    line_img = thresh_img[line[0]:line[1], :]
    
    # Sum up pixel values horizontally to find letters
    vertical_sum = np.sum(line_img, axis=0)
    
    # Find boundaries of letters
    letters = []
    letter_start = None
    threshold = 225  # Adjust threshold to control sensitivity

    for i, value in enumerate(vertical_sum):
        if value > threshold and letter_start is None:
            letter_start = i
        elif value < threshold and letter_start is not None:
            letters.append((letter_start, i))
            letter_start = None
            
    return letters, line_img

# Function to extract and save individual letters
def extract_letters(image_path, output_folder):
    # Preprocess the image
    thresh_img = preprocess_image(image_path)
    
    # Split image into lines
    print("Thresh img: ", thresh_img)
    lines = split_into_lines(thresh_img)
    # print(lines)
    letter_count = 0
    
    for idx, line in enumerate(lines):
        # Split each line into letters
        letters, line_img = split_line_into_letters(thresh_img, line)
        
        # for jdx, (start, end) in enumerate(letters):
        #     # Extract the letter image
        #     letter_img = line_img[:, start:end]
            
        #     # Save the letter image
        #     letter_filename = f"{output_folder}/letter_{idx}_{jdx}.png"
        #     cv2.imwrite(letter_filename, letter_img)
            
        #     letter_count += 1
        # if line_img.shape[0] > 20:
        cv2.imwrite(f"{output_folder}/letter_{idx}.png", line_img)
        letter_count += 1
            
    print(f"Extracted {letter_count} letters.")

# Example usage
image_path = 'l11.jpg'  # Path to your image containing text
output_folder = 'output_letters1'  # Folder where the extracted letters will be saved
f = open("lines.txt",'w')
# Call the function to extract letters
extract_letters(image_path, output_folder)
