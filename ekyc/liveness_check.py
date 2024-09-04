import cv2
import tempfile
import os
from deepface import DeepFace

def perform_liveness_check(image):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, image)
            
            face_objs = DeepFace.extract_faces(img_path=temp_file.name, anti_spoofing=True, enforce_detection=False)
            
            if len(face_objs) > 0:
                main_face = max(face_objs, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                is_live = main_face["is_real"] and main_face["confidence"] > 0.85
            else:
                is_live = False
            
            return is_live
    except Exception:
        return False
    finally:
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass