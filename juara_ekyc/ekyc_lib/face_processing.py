import cv2
import dlib
from .utils import setup_logger

logger = setup_logger(__name__)

# Load the face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def process_ic_face(image, ic_bbox):
    logger.info("Processing IC face")
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
    
    if len(faces) == 0:
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
    logger.info("Masking IC face")
    masked_image = image.copy()
    top, right, bottom, left = ic_face_location
    cv2.rectangle(masked_image, (left, top), (right, bottom), (0, 0, 0), -1)
    return masked_image