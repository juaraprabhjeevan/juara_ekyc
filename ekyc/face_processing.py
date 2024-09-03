import cv2
import dlib
import pkg_resources
from .utils import setup_logger

logger = setup_logger(__name__)

# Load the face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor_path = pkg_resources.resource_filename('ekyc', 'data/shape_predictor_68_face_landmarks.dat')
shape_predictor = dlib.shape_predictor(predictor_path)

def process_ic_face(image, ic_bbox):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)
    
    if len(faces) < 2:
        return None, None
    
    # Sort faces by area in descending order
    faces = sorted(faces, key=lambda f: (f.bottom()-f.top())*(f.right()-f.left()), reverse=True)
    
    # The second largest face is the IC face
    ic_face = faces[1]
    return image[ic_face.top():ic_face.bottom(), ic_face.left():ic_face.right()], (ic_face.top(), ic_face.right(), ic_face.bottom(), ic_face.left())

def mask_all_faces_except_largest(image):
    masked_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)
    
    if len(faces) == 0:
        return masked_image
    
    # Sort faces by area in descending order
    faces = sorted(faces, key=lambda f: (f.bottom()-f.top())*(f.right()-f.left()), reverse=True)
    
    # Mask all faces except the largest
    for face in faces[1:]:
        cv2.rectangle(masked_image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 0), -1)
    
    return masked_image