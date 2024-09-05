from .core import process_id_verification
from .face_processing import get_ic_face, mask_ic_and_smaller_faces, face_detector, shape_predictor
from .document_verification import verify_document
from .liveness_check import perform_liveness_check
from .face_matching import match_faces
from .utils import preprocess_image, get_base64_encoded_image, analyze_background_color, setup_logger

__all__ = [
    'process_id_verification',
    'verify_document',
    'get_ic_face',
    'mask_ic_and_smaller_faces',
    'face_detector',
    'shape_predictor',
    'perform_liveness_check',
    'match_faces',
    'preprocess_image',
    'get_base64_encoded_image',
    'analyze_background_color',
    'setup_logger'
]

__version__ = "0.1.15" 