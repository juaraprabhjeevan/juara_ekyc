import unittest
import os
import cv2
import numpy as np
from ekyc import process_id_verification

class TestEKYCLib(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample image for testing
        cls.test_image_path = 'test_id_image.jpg'
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(test_image, 'MYKAD', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.circle(test_image, (200, 300), 100, (255, 255, 255), -1)  # Large face
        cv2.circle(test_image, (600, 300), 50, (255, 255, 255), -1)  # Small face (IC face)
        cv2.imwrite(cls.test_image_path, test_image)

    @classmethod
    def tearDownClass(cls):
        # Clean up the test image
        os.remove(cls.test_image_path)

    def test_process_id_verification(self):
        result = process_id_verification(self.test_image_path)
        self.assertIsInstance(result, dict)
        self.assertIn('document_verification', result)
        self.assertIn('face_processing', result)
        self.assertIn('liveness_check', result)
        self.assertIn('face_matching', result)
        self.assertIn('final_result', result)
        self.assertIn('final_message', result)

    def test_process_id_verification_with_invalid_image(self):
        result = process_id_verification('non_existent_image.jpg')
        self.assertFalse(result['final_result'])

if __name__ == '__main__':
    unittest.main()