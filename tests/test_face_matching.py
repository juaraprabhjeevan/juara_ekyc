import unittest
import cv2
import numpy as np
from ekyc.face_matching import match_faces

class TestFaceMatching(unittest.TestCase):
    def setUp(self):
        # Create two sample face images for testing
        self.face1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.face2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_match_faces(self):
        match_result, explanation = match_faces(self.face1, self.face2)
        self.assertIsInstance(match_result, bool)
        self.assertIsInstance(explanation, str)

    def test_match_faces_with_same_image(self):
        match_result, explanation = match_faces(self.face1, self.face1)
        self.assertTrue(match_result)

    def test_match_faces_with_invalid_image(self):
        invalid_image = np.zeros((1, 1, 3), dtype=np.uint8)
        match_result, explanation = match_faces(invalid_image, self.face1)
        self.assertFalse(match_result)

if __name__ == '__main__':
    unittest.main()