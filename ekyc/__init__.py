from .core import process_id_verification
from .document_verification import verify_document
from .face_processing import process_ic_face, mask_ic_face, face_detector, shape_predictor
from .liveness_check import perform_liveness_check
from .face_matching import match_faces
from .utils import preprocess_image, get_base64_encoded_image, analyze_background_color, setup_logger

__all__ = [
    'process_id_verification',
    'verify_document',
    'process_ic_face',
    'mask_ic_face',
    'face_detector',
    'shape_predictor',
    'perform_liveness_check',
    'match_faces',
    'preprocess_image',
    'get_base64_encoded_image',
    'analyze_background_color',
    'setup_logger'
]

__version__ = "0.0.5"