import cv2
import tempfile
import os
from deepface import DeepFace
import logging
import time

logger = logging.getLogger(__name__)

def perform_liveness_check(image):
    temp_file = None
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        logger.info(f"Created temporary file: {temp_file.name}")

        # Write the image to the temporary file
        cv2.imwrite(temp_file.name, image)
        logger.info(f"Image written to temporary file")

        # Close the file to ensure it's not held open
        temp_file.close()

        # Perform face extraction with anti-spoofing
        face_objs = DeepFace.extract_faces(img_path=temp_file.name, anti_spoofing=True, enforce_detection=False)
        logger.info(f"Face extraction completed. Number of faces detected: {len(face_objs)}")

        if len(face_objs) > 0:
            main_face = max(face_objs, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
            is_live = main_face["is_real"] and main_face["confidence"] > 0.85
            logger.info(f"Liveness check result: is_live={is_live}, is_real={main_face['is_real']}, confidence={main_face['confidence']}")
        else:
            is_live = False
            logger.warning("No faces detected in the image")

        return is_live

    except Exception as e:
        logger.error(f"Error during liveness check: {str(e)}")
        return False
    finally:
        if temp_file:
            for _ in range(5):  # Try 5 times
                try:
                    os.unlink(temp_file.name)
                    logger.info(f"Temporary file deleted: {temp_file.name}")
                    break
                except Exception as e:
                    logger.warning(f"Error deleting temporary file: {str(e)}. Retrying...")
                    time.sleep(0.5)  # Wait for 500ms before retrying
            else:
                logger.error(f"Failed to delete temporary file after 5 attempts: {temp_file.name}")