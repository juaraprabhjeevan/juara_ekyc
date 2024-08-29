import cv2
import tempfile
import os
from deepface import DeepFace
from .utils import setup_logger

logger = setup_logger(__name__)

def match_faces(user_face_image, ic_face_image):
    logger.info("Matching faces")
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
            logger.error(f"Error in face matching: {str(e)}")
            return False, f"Error: {str(e)}"
        finally:
            os.unlink(user_file.name)
            os.unlink(ic_file.name)