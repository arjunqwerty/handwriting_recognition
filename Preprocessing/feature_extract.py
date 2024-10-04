import cv2
import os
import csv
import numpy as np
import shutil
from skimage.feature import hog
from skimage.morphology import skeletonize
from skimage import exposure

# Folder paths
annotation_csv = 'annotations.csv'
dataset_folder = 'dataset'  # Folder containing original images
output_folders = {
    'enhanced': 'feature_extraction\\1_enhanced_images',
    'segmented': 'feature_extraction\\2_segmented_images',
    'edges': 'feature_extraction\\3_edge_images',
    'skeleton': 'feature_extraction\\4_skeleton_images',
    'hog': 'feature_extraction\\5_hog_images'
}

if os.path.exists('feature_extraction'):
    shutil.rmtree('feature_extraction')
# Create output directories if they don't exist
for folder in output_folders.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load annotations
with open(annotation_csv, 'r') as file:
    reader = csv.reader(file)
    annotations = list(reader)[1:]  # Skip header   
    # annotations = list(reader)[2880:]  # Skip header
    # annotations = list(reader)[1:100]  # Skip header
    # annotations = list(reader)[2918:]  # Skip header

flag = annotations[0][0]
c = 0
count = 0
# Feature extraction steps
for annotation in annotations:
    c += 1
    image_name, letter, center_x, center_y, dist_x, dist_y = annotation
    if flag == image_name:
        count += 1
    else:
        count = 1
        flag = image_name

    # Load original image
    # image_path = os.path.join(dataset_folder, image_name)
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image_name = image_name[image_name.find('\\')+1:image_name.find('.')]
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: {image_name}_{c} not found!")
        continue

    # Crop the letter using the annotation box
    start_x = int(float(center_x) - float(dist_x) / 2)
    start_y = int(float(center_y) - float(dist_y) / 2)
    end_x = int(float(center_x) + float(dist_x) / 2)
    end_y = int(float(center_y) + float(dist_y) / 2)
    cropped = img[start_y:end_y, start_x:end_x]

    # 1. Image Enhancement & Normalization (Histogram Equalization)
    resized = cv2.resize(cropped, (64, 64))
    # denoised_image = cv2.bilateralFilter(resized, 9, 75, 75)
    # denoised_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)
    enhanced = cv2.equalizeHist(resized)
    # denoised_image = cv2.GaussianBlur(resized, (5, 5), 0)
    # enhanced = cv2.equalizeHist(denoised_image)
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
    # kernel = np.ones((2, 2), np.uint8)
    # skeleton = cv2.dilate(skeleton, kernel, iterations=1)
    skeleton_path = os.path.join(output_folders['skeleton'], f'{image_name}_{letter}_{count}.png')
    cv2.imwrite(skeleton_path, skeleton)

    # 5. HOG (Histogram of Oriented Gradients)
    hog_features, hog_image = hog(segmented, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    hog_features, hog_image = hog(segmented, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    # hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)
    # hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255))
    hog_path = os.path.join(output_folders['hog'], f'{image_name}_{letter}_{count}.png')
    cv2.imwrite(hog_path, hog_image)
    # cv2.imwrite(hog_path, hog_image_rescaled)

    # print(f"Processed {image_name} {letter} {count}")

print("Feature extraction completed for all annotated letters.")
