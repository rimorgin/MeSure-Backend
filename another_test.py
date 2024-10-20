# Import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from bgremover import BackgroundRemover
import os

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def removeBG(image, filepath):
    remover = BackgroundRemover(image)
    
    try:
        # Process the image and get the result as a PIL Image
        output_image = remover.process_image()
        
        # Check if the image has an alpha channel (RGBA)
        if output_image.mode == 'RGBA':
            output_image = output_image.convert('RGB')  # Convert to RGB if necessary
        
        # Save the output image using OpenCV
        output_image.save(filepath)

        # Load the image in OpenCV
        removedBG = cv2.imread(filepath)

        # Check if the image was loaded correctly
        if removedBG is not None:
            return removedBG  # return the loaded image
        else:
            print("Error: Could not load the image for display.")

    except Exception as e:
        print(e)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# Load the background-removed image
output_filename = os.path.splitext(args['image'])[0] + 'BG.jpg'
img = removeBG(args['image'], output_filename)

# Convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# Perform edge detection
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cv2.imshow('edged', edged)

# Find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#print(cnts)

# Check if any contours were found
if len(cnts) == 0:
    print("No contours found.")
else:
    # Get the contour with the max area (largest object)
    c = max(cnts, key=cv2.contourArea)

    # Approximate the contour to reduce its complexity
    epsilon = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # Get the convex hull of the simplified contour
    hull = cv2.convexHull(approx)
    

    # Draw the original contour and the convex hull on the image
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)  # Original contour in green
    cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)  # Convex hull in blue

    cv2.imshow("Convexity Hull", img)
   


    # Show the final image with the original contour, convex hull, and convexity defects
    #cv2.imshow("Convexity Defects", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
os.remove(output_filename)
