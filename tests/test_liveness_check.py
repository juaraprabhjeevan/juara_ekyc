import unittest
import cv2
import numpy as np
from ekyc.liveness_check import perform_liveness_check

class TestLivenessCheck(unittest.TestCase):
    def setUp(self):
        # Create a sample image for testing
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(self.test_image, (150, 150), 100, (255, 255, 255), -1)

    def test_perform_liveness_check(self):
        result = perform_liveness_check(self.test_image)
        self.assertIsInstance(result, bool)

    def test_perform_liveness_check_with_empty_image(self):
        empty_image = np.zeros((1, 1, 3), dtype=np.uint8)
        result = perform_liveness_check(empty_image)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()