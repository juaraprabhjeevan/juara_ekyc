import tempfile
import os
import base64
import cv2
import pkg_resources
from .document_verification import verify_document
from .face_processing import process_ic_face, mask_ic_face
from .liveness_check import perform_liveness_check
from .face_matching import match_faces
from .utils import preprocess_image, get_base64_encoded_image, setup_logger

logger = setup_logger(__name__)

def process_id_verification(image_path, template_path=None):
    if template_path is None:
        template_path = pkg_resources.resource_filename('ekyc', 'data/ic_template.jpg')
    
    logger.info("Starting ID verification process")
    TEMPLATE_IMAGE_BASE64 = get_base64_encoded_image(template_path)
    
    # Decode the embedded template image
    template_image_data = base64.b64decode(TEMPLATE_IMAGE_BASE64)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as template_temp:
        template_temp.write(template_image_data)
        template_temp.flush()
        
        is_verified, ic_bbox = verify_document(image_path, template_temp.name)
    
    os.unlink(template_temp.name)
    
    if not is_verified:
        logger.warning("Document verification failed")
        return False, "Document verification failed."
    
    image = preprocess_image(image_path)
    ic_face_image, ic_face_location = process_ic_face(image, ic_bbox)
    if ic_face_image is None or ic_face_location is None:
        logger.warning("Failed to detect a suitable face for IC")
        return False, "Failed to detect a suitable face for IC."
    
    masked_image = mask_ic_face(image, ic_face_location)
    is_live = perform_liveness_check(masked_image)
    if not is_live:
        logger.warning("Liveness check failed")
        return False, "Liveness check failed. The user's face might be spoofed."
    
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        logger.warning("No face detected in the image after masking IC")
        return False, "No face detected in the image after masking IC."
    
    (x, y, w, h) = faces[0]
    user_face_image = masked_image[y:y+h, x:x+w]
    
    faces_match, explanation = match_faces(user_face_image, ic_face_image)
    if faces_match:
        logger.info("Face matching successful")
        return True, "Face matching successful. Ready for API call."
    else:
        logger.warning(f"Face matching failed: {explanation}")
        return False, f"Face matching failed. {explanation}"