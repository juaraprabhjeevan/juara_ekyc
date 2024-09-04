import logging
import cv2
import base64
import numpy as np
from sklearn.cluster import KMeans
import os

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image  # Return the image as-is, without any color conversion

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_background_color(image):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=5, random_state=42).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    lower_blue = np.array([90, 120, 130])
    upper_blue = np.array([205, 225, 255])
    lower_gray_blue = np.array([180, 200, 200])
    upper_gray_blue = np.array([220, 240, 240])
    
    for color in colors:
        if np.all(color >= lower_blue) and np.all(color <= upper_blue):
            if color[2] >= color[0] and abs(color[2] - color[1]) <= 10:
                return True
        if np.all(color >= lower_gray_blue) and np.all(color <= upper_gray_blue):
            if max(color) - min(color) <= 20 and color[2] >= color[0]:
                return True
    return False

def log_image(image, filename):
    log_dir = "ekyc_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cv2.imwrite(os.path.join(log_dir, filename), image)