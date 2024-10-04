import cv2
import os
import csv
import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage.morphology import skeletonize

def display_image(img):
    global scale, dx, dy, annot_center_start_x, annot_center_start_y, annot_center_end_x, annot_center_end_y, annot_start_x1, annot_start_y1, annot_end_x1, annot_end_y1

    height, width = img.shape[:2]
    # Scale the image
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    scaled_img = cv2.resize(img, (scaled_width, scaled_height))

    # Adjust canvas for dragging
    canvas = scaled_img[max(dy, 0):min(scaled_height + dy, scaled_height), max(dx, 0):min(scaled_width + dx, scaled_width)]
    display_img = canvas[0:canvas_height, 0:canvas_width]

    # Draw the annot box (centered)
    annot_start_x1 = int(min(annot_center_start_x, annot_center_start_x+dx))
    annot_start_y1 = int(min(annot_center_start_y, annot_center_start_y+dy))
    annot_end_x1 = int(min(annot_center_end_x, annot_center_end_x+dx))
    annot_end_y1 = int(min(annot_center_end_y, annot_center_end_y+dy))

    cv2.rectangle(display_img, (annot_start_x1, annot_start_y1), (annot_end_x1, annot_end_y1), (0, 255, 0), 2)  # Green box
    cv2.putText(display_img, str(scale), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Annot Tool', display_img)

# Mouse callback function to handle dragging and zooming
def mouse_callback(event, x, y, flags, param):
    global dx, dy, start_x, start_y, is_dragging, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        is_dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        dx -= int((x - start_x)/scale)
        dy -= int((y - start_y)/scale)
        if dx < -annot_center_start_x: dx=-annot_center_start_x
        if dy < -annot_center_start_y: dy=-annot_center_start_y
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Scroll to zoom in or out, adjusting the scale factor
        if flags > 0:
            scale = min(scale + 0.05, 5.0)  # Zoom in, with a max scale
        else:
            scale = max(scale - 0.05, 0.1)  # Zoom out, with a minimum scale

# Function to confirm and save annotation details in a CSV
def confirm_and_save_annotation(img, key):
    global cnt
    image_name = images[current_image_idx]
    asy, aey, asx, aex = int(annot_start_y1/scale), int(annot_end_y1/scale), int(annot_start_x1/scale), int(annot_end_x1/scale)
    if annot_center_start_y==annot_start_y1:
        asy += int(dy/scale)
        aey += int(dy/scale)
    if annot_center_start_x==annot_start_x1:
        asx += int(dx/scale)
        aex += int(dx/scale)
    cropped = img[asy:aey, asx:aex]
    letter = chr(key)
    letter = chr(key)
    selected_letter = cv2.resize(cropped, (180, 180))
    cv2.imshow(f"{letter} Selected Image", selected_letter)
    while True:
        key1 = cv2.waitKey(1) & 0xFF
        if key1 == key:  # Enter key to confirm
            cnt += 1
            annotation_center_x = (asx + aex) // 2
            annotation_center_y = (asy + aey) // 2
            annotation_distance_x = aex - asx
            annotation_distance_y = aey - asy

            df = pd.read_csv(annotation_csv)
            count = len(df[df['Image'] == image_name])

            # Save to CSV file
            with open(annotation_csv, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([image_name, letter, annotation_center_x, annotation_center_y, annotation_distance_x, annotation_distance_y])
            print(f'Annotation saved for {image_name}_{letter}')

            image_name = image_name[image_name.find('\\')+1:image_name.find('.')]
            ### Feature Extraction
            # 1. Image Enhancement & Normalization (Histogram Equalization)
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            enhanced = cv2.equalizeHist(resized)
            norm_image = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            enhanced_path = os.path.join(output_folders['enhanced'], f'{image_name}_{letter}_{count}.png')
            cv2.imwrite(enhanced_path, norm_image)

            # 2. Segmentation (Thresholding)
            _, segmented = cv2.threshold(norm_image, 30, 255, cv2.THRESH_BINARY_INV)
            segmented_path = os.path.join(output_folders['segmented'], f'{image_name}_{letter}_{count}.png')
            cv2.imwrite(segmented_path, segmented)

            # 3. Edge Detection (Canny)
            edges = cv2.Canny(segmented, 100, 200)
            edges_path = os.path.join(output_folders['edges'], f'{image_name}_{letter}_{count}.png')
            cv2.imwrite(edges_path, edges)

            # 4. Skeletonization
            binary = segmented / 255  # Convert to binary (0, 1)
            skeleton = skeletonize(binary).astype(np.uint8) * 255
            skeleton_path = os.path.join(output_folders['skeleton'], f'{image_name}_{letter}_{count}.png')
            cv2.imwrite(skeleton_path, skeleton)

            # 5. HOG (Histogram of Oriented Gradients)
            _, hog_image = hog(segmented, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
            hog_path = os.path.join(output_folders['hog'], f'{image_name}_{letter}_{count}.png')
            cv2.imwrite(hog_path, hog_image)

            cv2.destroyWindow(f"{letter} Selected Image")
            break
        elif key1 == 27:  # Escape key to cancel
            cv2.destroyWindow(f"{letter} Selected Image")
            break

# Variables for zoom, drag, and dragging state
scale = 1.0
dx, dy = 0, 0
is_dragging = False
start_x, start_y = 0, 0
annot_box_height, annot_box_width = 70, 50
canvas_height, canvas_width = 800, 800
annot_center_start_x = (canvas_width - annot_box_width) // 2
annot_center_start_y = (canvas_height - annot_box_height) // 2
annot_center_end_x = annot_center_start_x + annot_box_width
annot_center_end_y = annot_center_start_y + annot_box_height
annot_start_x1, annot_start_y1, annot_end_x1, annot_end_y1 = 0, 0, 0, 0

# Paths
folder_path = 'dataset'
annotation_csv = 'annotations.csv'
output_folders = {
    'enhanced': 'feature_extraction\\1_enhanced_images',
    'segmented': 'feature_extraction\\2_segmented_images',
    'edges': 'feature_extraction\\3_edge_images',
    'skeleton': 'feature_extraction\\4_skeleton_images',
    'hog': 'feature_extraction\\5_hog_images'
}
# with open(annotation_csv, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(["Image", "Letter", "Center_x", "Center_y", "Distance_x", "Distance_y"])

# Load all images from the folder
images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.jpg') or img.endswith('.png')]
current_image_idx = 0

cnt = 0
# Main loop for annotation tool
while True:
    img = cv2.imread(images[current_image_idx])
    print(images[current_image_idx])
    cv2.namedWindow('Annot Tool')

    while True:
        cv2.setMouseCallback('Annot Tool', mouse_callback)
        display_image(img)
        curr_height, curr_width = img.shape[:2]
        if dx > ((curr_width+annot_center_start_x-canvas_width)*scale): dx = int(((curr_width+annot_center_start_x-canvas_width)*scale))
        if dy > ((curr_height+annot_center_start_y)*scale)-canvas_height: dy = int(((curr_height+annot_center_start_y)*scale)-canvas_height)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 'Esc' to quit
            cv2.destroyAllWindows()
            break
        elif key == 32:  # 'Space' for next image
            current_image_idx += 1
            if current_image_idx >= len(images):
                current_image_idx = 0
            break
        elif 65 <= key <= 90 or 97 <= key <= 122:
            confirm_and_save_annotation(img, key)
    if key == 27:
        break
print(cnt, "number of letters annotated")
