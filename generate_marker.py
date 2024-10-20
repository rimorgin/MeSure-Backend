import cv2
import numpy as np

# Set the dictionary to DICT_5X5_50
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# Define the marker ID, size, and image size
marker_id = 33  # You can choose any ID between 0 and 49, as this dictionary supports 50 markers
marker_size = 200  # Marker size in pixels

# Create an image to draw the marker on
marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)

# Generate the marker image
cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size, marker_image)

# Save the marker image
cv2.imwrite('aruco_marker_5x5_50_id33.png', marker_image)

# Display the marker
cv2.imshow('ArUco Marker', marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
