import cv2
import tempfile
import os
import time
from deepface import DeepFace
from .utils import setup_logger

logger = setup_logger(__name__)

def match_faces(user_face_image, ic_face_image):
    logger.info("Matching faces")
    
    user_file = None
    ic_file = None
    
    try:
        user_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        ic_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        
        cv2.imwrite(user_file.name, user_face_image)
        cv2.imwrite(ic_file.name, ic_face_image)
        
        result = DeepFace.verify(img1_path=user_file.name, img2_path=ic_file.name, enforce_detection=False)
        distance = result['distance']
        threshold = 0.75
        match_result = distance < threshold
        explanation = f"Faces matched with a distance of {distance:.4f}."
        return match_result, explanation
    except Exception as e:
        logger.error(f"Error during face matching: {str(e)}")
        return False, f"Face matching failed: {str(e)}"
    finally:
        if user_file:
            user_file.close()
        if ic_file:
            ic_file.close()
        
        # Attempt to remove temporary files with retry
        for file in [user_file, ic_file]:
            if file:
                for _ in range(5):  # Try 5 times
                    try:
                        os.unlink(file.name)
                        break
                    except PermissionError:
                        time.sleep(0.1)  # Wait a bit before retrying
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {file.name}: {str(e)}")
                        break