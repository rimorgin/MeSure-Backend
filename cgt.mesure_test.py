from imutils import contours
import argparse
import imutils
import cv2
from bgremover import BackgroundRemover
from handtracker import HandImageProcessor
from calsize import BoundingBoxAnalyzer
import os
from PIL import Image
import numpy as np


# Global variable for pixels per metric
pixelsPerMetric = None

def removeBG(image, filepath):
    input_image = Image.open(image)
    remover = BackgroundRemover(input_image)
    
    try:
        # Process the image and get the result as a PIL Image
        output_image = remover.process_image()
        
        if output_image is None:
            print("Error: Background removal failed, output_image is None.")
            return None
        
        # Check if the image has an alpha channel (RGBA)
        if output_image.mode == 'RGBA':
            output_image = output_image.convert('RGB')  # Convert to RGB if necessary
        
        # Save the output image using OpenCV
        output_image.save(filepath)

        # Convert the PIL image to a NumPy array
        output_image = np.array(output_image)  # Convert to NumPy array

        return output_image  # Return the NumPy array

    except Exception as e:
        print(e)
        
# Function to track fingers for isolation
def trackFinger(image):
    processor = HandImageProcessor()
    processed_image, _ = processor.process_hand_image(image)  # Ensure this returns a NumPy array
    return processed_image  # Return the processed image directly

def process_image(image):
    # Convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    # Perform edge detection
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Check if any contours were found
    if not cnts:
        print("No contours found.")
        return []  # Return an empty list if no contours are found

    # Sort contours from top to bottom
    cnts, _ = contours.sort_contours(cnts, method='top-to-bottom')

    return cnts

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the reference object in mm")
args = vars(ap.parse_args())

# Load the original image
orig_image = cv2.imread(args["image"])
reference_width = args["width"]

# Process the original image
reference_cnts = process_image(orig_image)
if reference_cnts:
    calSize = BoundingBoxAnalyzer(orig_image, args["width"])
    reference_cnt = reference_cnts[0]  # Reference object
    calSize.cal_reference_size(reference_cnt)
    cv2.imshow('orig image', orig_image)
    cv2.waitKey(0)

# Load the background-removed image
output_filename = os.path.splitext(args['image'])[0] + 'BG.jpg'
noBG = removeBG(args["image"], output_filename)

if noBG is not None:
    cv2.imshow('noBG', noBG)
    noBG = trackFinger(noBG)
    cv2.imshow('figner', noBG)
    cv2.waitKey(0)
else:
    print("Error: No background-removed image to process.")
    exit()

# Process the background-removed image
finger_cnts = process_image(noBG)
if finger_cnts:
    hand_cnts = finger_cnts[:5]  # Skip the first contour if it's the reference object
    hand_cnts, _ = contours.sort_contours(hand_cnts, method="top-to-bottom")

    # Filter small contours
    min_area = 1000
    finger_cnts = [c for c in hand_cnts if cv2.contourArea(c) > min_area]

    # Process each finger contour
    for (i, c) in enumerate(finger_cnts):
        M = cv2.moments(c)
        if M["m00"] != 0:  # Avoid division by zero
            calSize.cal_finger_size(c)  # Calculate size if needed

    # Show the final image with contours and labels
    cv2.imshow("Background Removed and Finger Tracking", noBG)
    cv2.imshow("Calculated Size", orig_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

os.remove(output_filename)
