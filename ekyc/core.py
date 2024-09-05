import tempfile
import os
import base64
import cv2
import pkg_resources
from .document_verification import verify_document
from .liveness_check import perform_liveness_check
from .face_processing import get_ic_face, mask_ic_and_smaller_faces, detect_face, preprocessImage
from .face_matching import match_faces
from .utils import preprocess_image, get_base64_encoded_image
import logging
import traceback

logger = logging.getLogger(__name__)

def process_id_verification(image_path, template_path=None):
    try:
        if template_path is None:
            template_path = pkg_resources.resource_filename('ekyc', 'data/ic_template.jpg')
        
        TEMPLATE_IMAGE_BASE64 = get_base64_encoded_image(template_path)
        
        template_image_data = base64.b64decode(TEMPLATE_IMAGE_BASE64)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as template_temp:
            template_temp.write(template_image_data)
            template_temp.flush()
            
            is_verified, ic_bbox = verify_document(image_path, template_temp.name)
        
        os.unlink(template_temp.name)
        
        results = {
            "document_verification": {
                "is_verified": is_verified,
                "ic_bbox": ic_bbox
            }
        }

        if not is_verified:
            results["final_result"] = False
            results["final_message"] = "Document verification failed."
            return results

        image = preprocess_image(image_path)

        ic_face_image, ic_face_bbox = get_ic_face(image)
        results["face_processing"] = {
            "ic_face_detected": ic_face_image is not None and ic_face_bbox is not None,
            "ic_face_bbox": ic_face_bbox
        }

        if ic_face_image is None or ic_face_bbox is None:
            results["final_result"] = False
            results["final_message"] = "Failed to detect a suitable face for IC."
            return results

        masked_image = mask_ic_and_smaller_faces(image)

        is_live = perform_liveness_check(masked_image)
        results["liveness_check"] = {
            "is_live": is_live
        }

        if not is_live:
            results["final_result"] = False
            results["final_message"] = "Liveness check failed. The user's face might be spoofed."
            logger.warning("Liveness check failed")
            return results
        
        user_face_image = detect_face(masked_image)
        results["face_processing"]["user_face_detected"] = user_face_image is not None

        if user_face_image is None:
            results["final_result"] = False
            results["final_message"] = "No face detected in the image after masking IC."
            return results
        
        user_face_image = preprocessImage(user_face_image)
        ic_face_image = preprocessImage(ic_face_image)
        
        faces_match, explanation = match_faces(user_face_image, ic_face_image)
        results["face_matching"] = {
            "faces_match": faces_match,
            "explanation": explanation
        }

        if faces_match:
            results["final_result"] = True
            results["final_message"] = "Face matching successful. Ready for API call."
        else:
            results["final_result"] = False
            results["final_message"] = f"Face matching failed. {explanation}"

        return results
    except Exception as e:
        logger.error(f"Unexpected error in process_id_verification: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "final_result": False,
            "final_message": "An unexpected error occurred during ID verification.",
            "error_details": str(e)
        }