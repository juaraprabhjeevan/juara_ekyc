import unittest
import os
import cv2
import numpy as np
from ekyc_lib.document_verification import verify_document
from ekyc_lib.utils import get_base64_encoded_image

class TestDocumentVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample image and template for testing
        cls.image_path = 'test_image.jpg'
        cls.template_path = 'test_template.jpg'
        
        # Create a simple test image
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(test_image, 'MYKAD', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(cls.image_path, test_image)
        
        # Create a simple test template
        template_image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(template_image, (10, 10), (390, 290), (255, 255, 255), 2)
        cv2.imwrite(cls.template_path, template_image)

    @classmethod
    def tearDownClass(cls):
        # Clean up the test images
        os.remove(cls.image_path)
        os.remove(cls.template_path)

    def test_verify_document(self):
        result, bbox = verify_document(self.image_path, self.template_path)
        self.assertIsInstance(result, bool)
        if result:
            self.assertIsInstance(bbox, tuple)
            self.assertEqual(len(bbox), 4)
        else:
            self.assertIsNone(bbox)

    def test_verify_document_with_invalid_image(self):
        result, bbox = verify_document('non_existent_image.jpg', self.template_path)
        self.assertFalse(result)
        self.assertIsNone(bbox)

    def test_verify_document_with_invalid_template(self):
        result, bbox = verify_document(self.image_path, 'non_existent_template.jpg')
        self.assertFalse(result)
        self.assertIsNone(bbox)

    def test_get_base64_encoded_image(self):
        base64_string = get_base64_encoded_image(self.image_path)
        self.assertIsInstance(base64_string, str)
        self.assertTrue(len(base64_string) > 0)

if __name__ == '__main__':
    unittest.main()