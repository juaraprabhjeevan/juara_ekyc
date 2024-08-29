from flask import Flask, request, jsonify
import logging
import tempfile
import os
import base64
import cv2
from ekyc_sdk import (
    preprocess_image,
    verify_document,
    process_ic_face,
    mask_ic_face,
    perform_liveness_check,
    match_faces,
    get_base64_encoded_image
)
from ekyc_sdk.face_processing import face_detector

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Generate the base64 string from your template image
template_path = 'ic_template.jpg'
TEMPLATE_IMAGE_BASE64 = get_base64_encoded_image(template_path)

# Print the first 100 characters of the base64 string to verify it's not empty
logger.info(f"Base64 string (first 100 chars): {TEMPLATE_IMAGE_BASE64[:100]}")

def process_id_verification(image_path):
    # Decode the embedded template image
    template_image_data = base64.b64decode(TEMPLATE_IMAGE_BASE64)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as template_temp:
        template_temp.write(template_image_data)
        template_temp.flush()
        
        is_verified, ic_bbox = verify_document(image_path, template_temp.name)
    
    os.unlink(template_temp.name)
    
    if not is_verified:
        return False, "Document verification failed."
    
    image = preprocess_image(image_path)
    ic_face_image, ic_face_location = process_ic_face(image, ic_bbox)
    if ic_face_image is None or ic_face_location is None:
        return False, "Failed to detect a suitable face for IC."
    
    masked_image = mask_ic_face(image, ic_face_location)
    is_live = perform_liveness_check(masked_image)
    if not is_live:
        return False, "Liveness check failed. The user's face might be spoofed."
    
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)
    if not faces:
        return False, "No face detected in the image after masking IC."
    
    face = faces[0]
    user_face_image = masked_image[face.top():face.bottom(), face.left():face.right()]
    
    faces_match, explanation = match_faces(user_face_image, ic_face_image)
    if faces_match:
        return True, "Face matching successful. Ready for API call."
    else:
        return False, f"Face matching failed. {explanation}"

@app.route('/verify_id', methods=['POST'])
def verify_id():
    logger.info("Received request to /verify_id")
    try:
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'Image file is required'}), 400
        
        image_file = request.files['image']
        logger.info(f"Received image file: {image_file.filename}")
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as image_temp:
            image_file.save(image_temp.name)
            logger.info(f"Saved image to temporary file: {image_temp.name}")
            
            try:
                result, message = process_id_verification(image_temp.name)
                logger.info(f"ID verification result: {result}, message: {message}")
                return jsonify({'verified': result, 'message': message})
            except Exception as e:
                logger.error(f"Error during ID verification: {str(e)}", exc_info=True)
                return jsonify({'error': str(e)}), 500
            finally:
                os.unlink(image_temp.name)
                logger.info(f"Deleted temporary file: {image_temp.name}")
    except Exception as e:
        logger.error(f"Unexpected error in verify_id route: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/test', methods=['GET'])
def test():
    logger.info("Received request to /test")
    return jsonify({'message': 'Test successful', 'template_image_length': len(TEMPLATE_IMAGE_BASE64)}), 200

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=False, host='0.0.0.0', port=5000)