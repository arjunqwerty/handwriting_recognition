import cv2
import numpy as np

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    return thresh, img.shape, img

# Function to split the image into lines
def split_into_lines(thresh_img):
    # Sum up pixel values vertically to find text lines
    horizontal_sum = np.sum(thresh_img, axis=1)
    
    # Find boundaries of lines
    lines = []
    line_start = None
    threshold = 255*5  # Adjust threshold to control sensitivity

    for i, value in enumerate(horizontal_sum):
        if value > threshold and line_start is None:
            line_start = i
        elif value < threshold and line_start is not None:
            lines.append((line_start, i))
            line_start = None
    # return lines
    lines1 = []
    i = 1
    while i < len(lines):
        if lines[i][0]-lines[i-1][1] < 10:
            lines1.append([lines[i-1][0], lines[i][1]])
            i += 2
        else:
            lines1.append(lines[i])
            i += 1
    return lines1

# Function to visualize split lines by pasting them on a black background
def visualize_split_lines(original_img, thresh_img, lines, output_path):
    # Create a black image (3-channel to accommodate red borders)
    visual_image = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)

    # Iterate over detected lines
    for line in lines:
        # Extract the line from the original image
        line_img = original_img[line[0]:line[1], :]
        line_thresh_img = thresh_img[line[0]:line[1], :]

        # Convert the single-channel binary image to 3 channels to paste
        line_thresh_color = cv2.cvtColor(line_thresh_img, cv2.COLOR_GRAY2BGR)

        # Paste the line onto the black background
        visual_image[line[0]:line[1], :] = line_thresh_color

        # Draw a red border around the line (on the black image with the pasted content)
        cv2.rectangle(visual_image, (0, line[0]), (original_img.shape[1], line[1]), (0, 0, 255), 2)
        cv2.putText(visual_image, str(line[0])+", "+str(line[1]), (10, line[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA, False)

    # Save the visualization
    cv2.imwrite(output_path, visual_image)
    print(f"Split lines visual saved at {output_path}")

# Main function to preprocess, split, and visualize lines
def extract_and_visualize_lines(image_path, output_folder, visualization_path):
    # Preprocess the image
    thresh_img, original_shape, original_img = preprocess_image(image_path)
    
    # Split image into lines
    lines = split_into_lines(thresh_img)

    # Save the extracted lines as images (optional, commenting out as not needed now)
    letter_count = 0
    for idx, line in enumerate(lines):
        # Save each line image if needed (commented out)
        # line_img = thresh_img[line[0]:line[1], :]
        # cv2.imwrite(f"{output_folder}/line_{idx}.png", line_img)
        letter_count += 1

    print(f"Extracted {letter_count} lines.")

    # Visualize the split lines by pasting them on a black background
    visualize_split_lines(original_img, thresh_img, lines, visualization_path)

# Example usage
image_path = 'l11.jpg'  # Path to your image containing text
output_folder = 'output_letters_4'  # Folder where the extracted lines will be saved (optional)
visualization_path = 'split.png'  # Path to save the visualized image

# Call the function to extract and visualize lines
extract_and_visualize_lines(image_path, output_folder, visualization_path)
