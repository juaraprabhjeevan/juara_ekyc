import numpy as np
import dlib
from deepface import DeepFace
from paddleocr import PaddleOCR
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify
import os
import tempfile
import base64
import logging
import cv2

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Generate the base64 string from your template image
template_path = 'ic_template.jpg'
TEMPLATE_IMAGE_BASE64 = get_base64_encoded_image(template_path)

# Print the first 100 characters of the base64 string to verify it's not empty
logger.info(f"Base64 string (first 100 chars): {TEMPLATE_IMAGE_BASE64[:100]}")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

def process_ic_face(image, ic_bbox):
    x, y, w, h = ic_bbox
    x, y = max(0, x), max(0, y)
    w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return None, None
    
    ic_roi = image[y:y+h, x:x+w]
    if ic_roi.size == 0:
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)
    if not faces:
        return None, None
    
    if len(faces) == 1:
        face = faces[0]
        if (x < face.left() < x+w) and (y < face.top() < y+h):
            return image[face.top():face.bottom(), face.left():face.right()], (face.top(), face.right(), face.bottom(), face.left())
        else:
            return None, None
    
    faces = sorted(faces, key=lambda f: (f.bottom()-f.top())*(f.right()-f.left()))
    for face in faces:
        if (x < face.left() < x+w or x < face.right() < x+w) and (y < face.top() < y+h or y < face.bottom() < y+h):
            return image[face.top():face.bottom(), face.left():face.right()], (face.top(), face.right(), face.bottom(), face.left())
    
    face = faces[0]
    return image[face.top():face.bottom(), face.left():face.right()], (face.top(), face.right(), face.bottom(), face.left())

def mask_ic_face(image, ic_face_location):
    masked_image = image.copy()
    top, right, bottom, left = ic_face_location
    cv2.rectangle(masked_image, (left, top), (right, bottom), (0, 0, 0), -1)
    return masked_image

def perform_liveness_check(image):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        try:
            face_objs = DeepFace.extract_faces(img_path=temp_file.name, anti_spoofing=True, enforce_detection=False)
            is_real = all(face_obj.get("is_real", False) for face_obj in face_objs)
            return is_real
        except Exception:
            return False
        finally:
            os.unlink(temp_file.name)

def match_faces(user_face_image, ic_face_image):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as user_file, \
         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as ic_file:
        cv2.imwrite(user_file.name, cv2.cvtColor(user_face_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(ic_file.name, cv2.cvtColor(ic_face_image, cv2.COLOR_RGB2BGR))
        try:
            result = DeepFace.verify(img1_path=user_file.name, img2_path=ic_file.name, enforce_detection=False)
            distance = result['distance']
            threshold = 0.75
            match_result = distance < threshold
            explanation = f"Faces matched with a distance of {distance:.4f}."
            return match_result, explanation
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            os.unlink(user_file.name)
            os.unlink(ic_file.name)

def process_id_verification(image_path):
    # Decode the embedded template image
    template_image_data = base64.b64decode(TEMPLATE_IMAGE_BASE64)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as template_temp:
        template_temp.write(template_image_data)
        template_temp.flush()
        
        is_verified, ic_bbox = verify_document(image_path, template_temp.name)
    
    os.unlink(template_temp.name)
    
    if not is_verified:
        return False, "Document verification failed."
    
    image = preprocess_image(image_path)
    ic_face_image, ic_face_location = process_ic_face(image, ic_bbox)
    if ic_face_image is None or ic_face_location is None:
        return False, "Failed to detect a suitable face for IC."
    
    masked_image = mask_ic_face(image, ic_face_location)
    is_live = perform_liveness_check(masked_image)
    if not is_live:
        return False, "Liveness check failed. The user's face might be spoofed."
    
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)
    if not faces:
        return False, "No face detected in the image after masking IC."
    
    face = faces[0]
    user_face_image = masked_image[face.top():face.bottom(), face.left():face.right()]
    
    faces_match, explanation = match_faces(user_face_image, ic_face_image)
    if faces_match:
        return True, "Face matching successful. Ready for API call."
    else:
        return False, f"Face matching failed. {explanation}"

@app.route('/verify_id', methods=['POST'])
def verify_id():
    logger.info("Received request to /verify_id")
    try:
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'Image file is required'}), 400
        
        image_file = request.files['image']
        logger.info(f"Received image file: {image_file.filename}")
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as image_temp:
            image_file.save(image_temp.name)
            logger.info(f"Saved image to temporary file: {image_temp.name}")
            
            try:
                result, message = process_id_verification(image_temp.name)
                logger.info(f"ID verification result: {result}, message: {message}")
                return jsonify({'verified': result, 'message': message})
            except Exception as e:
                logger.error(f"Error during ID verification: {str(e)}", exc_info=True)
                return jsonify({'error': str(e)}), 500
            finally:
                os.unlink(image_temp.name)
                logger.info(f"Deleted temporary file: {image_temp.name}")
    except Exception as e:
        logger.error(f"Unexpected error in verify_id route: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/test', methods=['GET'])
def test():
    logger.info("Received request to /test")
    return jsonify({'message': 'Test successful', 'template_image_length': len(TEMPLATE_IMAGE_BASE64)}), 200

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=False, host='0.0.0.0', port=5000)