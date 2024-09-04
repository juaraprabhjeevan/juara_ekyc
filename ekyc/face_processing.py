import cv2
import dlib
import numpy as np
import pkg_resources

face_detector = dlib.get_frontal_face_detector()
predictor_path = pkg_resources.resource_filename('ekyc', 'data/shape_predictor_68_face_landmarks.dat')
shape_predictor = dlib.shape_predictor(predictor_path)

def detect_and_sort_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    return sorted(faces, key=lambda f: (f.bottom()-f.top())*(f.right()-f.left()), reverse=True)

def get_ic_face(image):
    sorted_faces = detect_and_sort_faces(image)
    
    if len(sorted_faces) < 2:
        return None, None
    
    ic_face = sorted_faces[1]
    ic_face_image = image[ic_face.top():ic_face.bottom(), ic_face.left():ic_face.right()]
    return ic_face_image, (ic_face.left(), ic_face.top(), ic_face.right() - ic_face.left(), ic_face.bottom() - ic_face.top())

def mask_ic_and_smaller_faces(image):
    masked_image = image.copy()
    sorted_faces = detect_and_sort_faces(image)
    
    if len(sorted_faces) < 2:
        return masked_image
    
    for face in sorted_faces[1:]:
        cv2.rectangle(masked_image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), -1)
    
    return masked_image

def preprocessImage(image):
    return cv2.resize(image, (224, 224))

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) > 0:
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = image[y:y+h, x:x+w]
        return face_image
    else:
        return None