import cv2
import tempfile
import os
from deepface import DeepFace
from .utils import setup_logger

logger = setup_logger(__name__)

def perform_liveness_check(image):
    logger.info("Performing liveness check")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        cv2.imwrite(temp_file.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        try:
            face_objs = DeepFace.extract_faces(img_path=temp_file.name, anti_spoofing=True, enforce_detection=False)
            is_real = all(face_obj.get("is_real", False) for face_obj in face_objs)
            return is_real
        except Exception as e:
            logger.error(f"Error in liveness check: {str(e)}")
            return False
        finally:
            os.unlink(temp_file.name)