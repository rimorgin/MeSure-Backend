import cv2
from object_detector import *
import numpy as np

# Load Aruco detector
#parameters = cv2.aruco.DetectorParameters()
#aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


# Load Object Detector
detector = HomogeneousBgDetector()

# Load Image
img = cv2.imread("sample2.jpg")

# Get Aruco marker
#corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Draw polygon around the marker
#int_corners = np.int32(corners[0][0])  # Change to int32
#cv2.polylines(img, [int_corners], True, (0, 255, 0), 5)

# Aruco Perimeter
#aruco_perimeter = cv2.arcLength(corners[0][0], True)

# Pixel to cm ratio
#pixel_cm_ratio = aruco_perimeter / 20

contours = detector.detect_objects(img)

# Draw objects boundaries
#


cv2.imshow("Image", img)
cv2.waitKey(0)