import cv2
import numpy as np
from paddleocr import PaddleOCR
from .utils import analyze_background_color

def verify_document(image_path, template_path):
    def preprocess_image_for_verification(image_path):
        return cv2.imread(image_path, 0)
    
    def feature_matching(image, template):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image, None)
        kp2, des2 = orb.detectAndCompute(template, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance), kp1, kp2
    
    image = preprocess_image_for_verification(image_path)
    template = preprocess_image_for_verification(template_path)
    matches, kp1, kp2 = feature_matching(image, template)
    
    template_verified = len(matches) > 85
    if not template_verified:
        return False, None

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    color_image = cv2.imread(image_path)
    ocr_result = ocr.ocr(image_path, cls=True)
    
    keywords = ["KAD PENGENALAN", "MYKAD"]
    found_keywords = [keyword for line in ocr_result[0] for keyword in keywords if keyword in line[1][0].upper()]
    
    score = 0.8 if any(keyword in found_keywords for keyword in ["MYKAD", "KAD PENGENALAN"]) else 0
    if analyze_background_color(color_image):
        score += 0.2
    
    is_verified = template_verified and score > 0.75
    
    if is_verified:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        x, y, w, h = cv2.boundingRect(dst)
        
        image_h, image_w = image.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, image_w - x), min(h, image_h - y)
        
        return is_verified, (x, y, w, h)
    else:
        return is_verified, None