import unittest
import io
import base64
import cv2
from flask import Flask
import numpy as np
from cgt_mesure_api_dev import app  # Assuming your Flask app is in 'app.py'

class TestApp(unittest.TestCase):

    def setUp(self):
        """Setup for all tests, create a test client"""
        self.app = app.test_client()
        self.app.testing = True

    def create_image(self, width=500, height=500, color=(255, 255, 255)):
        """Create a simple image for testing"""
        # Create a white image
        image = np.full((height, width, 3), color, dtype=np.uint8)
        return image

    def test_measure_fingers_valid(self):
        """Test /measure-fingers with a valid image and reference width"""
        image = self.create_image()  # Create a white image
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        data = {
            'image': (io.BytesIO(img_bytes), 'test.jpg'),
            'width': '10'  # Reference width for measurement
        }

        response = self.app.post('/measure-fingers', data=data)
        self.assertEqual(response.status_code, 200)

        response_json = response.get_json()
        self.assertIn('hand_label', response_json)
        self.assertIn('finger_measurement', response_json)
        self.assertIn('processed_image', response_json)

    def test_measure_fingers_invalid_no_image(self):
        """Test /measure-fingers with no image file"""
        data = {'width': '10'}
        response = self.app.post('/measure-fingers', data=data)
        self.assertEqual(response.status_code, 400)
        response_json = response.get_json()
        self.assertEqual(response_json['error'], 'No image file or width provided')

    def test_measure_fingers_invalid_no_width(self):
        """Test /measure-fingers with no width provided"""
        image = self.create_image()  # Create a white image
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        data = {
            'image': (io.BytesIO(img_bytes), 'test.jpg')
        }

        response = self.app.post('/measure-fingers', data=data)
        self.assertEqual(response.status_code, 400)
        response_json = response.get_json()
        self.assertEqual(response_json['error'], 'No image file or width provided')

    def test_measure_wrist_valid(self):
        """Test /measure-wrist with a valid image and reference width"""
        image = self.create_image()  # Create a white image
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        data = {
            'image': (io.BytesIO(img_bytes), 'test.jpg'),
            'width': '15'  # Reference width for measurement
        }

        response = self.app.post('/measure-wrist', data=data)
        self.assertEqual(response.status_code, 200)

        response_json = response.get_json()
        self.assertIn('hand_label', response_json)
        self.assertIn('wrist_measurement', response_json)
        self.assertIn('processed_image', response_json)

    def test_measure_wrist_invalid_no_image(self):
        """Test /measure-wrist with no image file"""
        data = {'width': '15'}
        response = self.app.post('/measure-wrist', data=data)
        self.assertEqual(response.status_code, 400)
        response_json = response.get_json()
        self.assertEqual(response_json['error'], 'No image file or reference width provided')

    def test_health_check(self):
        """Test /healthz endpoint"""
        response = self.app.get('/healthz')
        self.assertEqual(response.status_code, 200)
        response_json = response.get_json()
        self.assertEqual(response_json['status'], 'ok')

if __name__ == '__main__':
    unittest.main()
