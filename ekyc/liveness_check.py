import cv2
import tempfile
import os
from deepface import DeepFace
from .utils import setup_logger
from .face_processing import mask_all_faces_except_largest

logger = setup_logger(__name__)

def perform_liveness_check(image):
    try:
        masked_image = mask_all_faces_except_largest(image)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, masked_image)
            
            face_objs = DeepFace.extract_faces(img_path=temp_file.name, anti_spoofing=True)
            is_live = all(face_obj["is_real"] for face_obj in face_objs)
            return is_live
    except Exception as e:
        logger.error(f"Error during liveness check: {str(e)}")
        return False
    finally:
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_file.name}: {str(e)}")