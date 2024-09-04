import unittest
import cv2
import numpy as np
from ekyc.face_processing import detect_and_sort_faces, get_ic_face, mask_ic_and_smaller_faces, preprocessImage, detect_face

class TestFaceProcessing(unittest.TestCase):
    def setUp(self):
        # Create a sample image with two faces for testing
        self.test_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(self.test_image, (150, 150), 50, (255, 255, 255), -1)  # Larger face
        cv2.circle(self.test_image, (450, 150), 30, (255, 255, 255), -1)  # Smaller face

    def test_detect_and_sort_faces(self):
        faces = detect_and_sort_faces(self.test_image)
        self.assertEqual(len(faces), 2)
        self.assertGreater(faces[0].area(), faces[1].area())

    def test_get_ic_face(self):
        ic_face, ic_bbox = get_ic_face(self.test_image)
        self.assertIsNotNone(ic_face)
        self.assertIsNotNone(ic_bbox)
        self.assertEqual(len(ic_bbox), 4)

    def test_mask_ic_and_smaller_faces(self):
        masked_image = mask_ic_and_smaller_faces(self.test_image)
        self.assertEqual(masked_image.shape, self.test_image.shape)

    def test_preprocessImage(self):
        processed_image = preprocessImage(self.test_image)
        self.assertEqual(processed_image.shape, (224, 224, 3))

    def test_detect_face(self):
        face = detect_face(self.test_image)
        self.assertIsNotNone(face)

if __name__ == '__main__':
    unittest.main()