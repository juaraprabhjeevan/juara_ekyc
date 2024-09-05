import cv2
import tempfile
import os
import time
from deepface import DeepFace
import logging

logger = logging.getLogger(__name__)

def match_faces(user_face_image, ic_face_image):
    user_file = ic_file = None
    try:
        user_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        ic_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        
        cv2.imwrite(user_file.name, user_face_image)
        cv2.imwrite(ic_file.name, ic_face_image)
        
        # Close files immediately after writing
        user_file.close()
        ic_file.close()
        
        result = DeepFace.verify(img1_path=user_file.name, img2_path=ic_file.name, enforce_detection=False)
        distance = result['distance']
        threshold = 0.75
        match_result = distance < threshold
        explanation = f"Faces matched with a distance of {distance:.4f}."
        return match_result, explanation
    except Exception as e:
        logger.error(f"Face matching failed: {str(e)}")
        return False, f"Face matching failed: {str(e)}"
    finally:
        for file in [user_file, ic_file]:
            if file:
                for _ in range(5):
                    try:
                        os.unlink(file.name)
                        logger.info(f"Temporary file deleted: {file.name}")
                        break
                    except Exception as e:
                        logger.warning(f"Error deleting temporary file: {str(e)}. Retrying...")
                        time.sleep(0.5)
                else:
                    logger.error(f"Failed to delete temporary file after 5 attempts: {file.name}")