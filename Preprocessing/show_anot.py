import cv2
import os
import csv
import pandas as pd

scale = 1.0
dx, dy = 0, 0
is_dragging = False
start_x, start_y = 0, 0
annot_box_height, annot_box_width = 70, 50

folder_path = 'dataset'
csv_file_path = 'annotations.csv'

images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.jpg') or img.endswith('.png')]
current_image_idx = 0

def display_image(img, img_name):
    global scale, dx, dy
    df = pd.read_csv(csv_file_path)
    img_anot = df[df['Image']==img_name]
    height, width = img.shape[:2]
    scaled_height, scaled_width = int(height * scale), int(width * scale)
    for i in range(len(img_anot)):
        xc, yc, xd, yd = img_anot.iloc[i]['Center_x'], img_anot.iloc[i]['Center_y'], img_anot.iloc[i]['Distance_x'], img_anot.iloc[i]['Distance_y']
        asx, aex, asy, aey = xc - int(xd/2), xc + int(xd/2), yc - int(yd/2), yc + int(yd/2)
        cv2.rectangle(img, (asx, asy), (aex, aey), (0, 0, 255), 1)
    scaled_img = cv2.resize(img, (scaled_width, scaled_height))
    display_img = scaled_img[max(dy, 0):min(scaled_height + dy, scaled_height), max(dx, 0):min(scaled_width + dx, scaled_width)]
    cv2.imshow('Annot Tool', display_img)

def mouse_callback(event, x, y, flags, param):
    global dx, dy, start_x, start_y, is_dragging, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        is_dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        dx -= int((x - start_x)/scale)
        dy -= int((y - start_y)/scale)
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scale = min(scale + 0.05, 5.0)
        else:
            scale = max(scale - 0.05, 0.1)

while True:
    img = cv2.imread(images[current_image_idx])
    print(images[current_image_idx])
    height, width = img.shape[:2]
    scale = 1000/height
    scaled_width = int(width * scale)
    scaled_height = 1000
    cv2.namedWindow('Annot Tool')
    while True:
        display_image(img, images[current_image_idx])
        cv2.setMouseCallback('Annot Tool', mouse_callback)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            exit()
        elif key == 32:
            current_image_idx += 1
            if current_image_idx >= len(images):
                current_image_idx = 0
            break
