from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image
from bgremover import BackgroundRemover
from handtracker import HandImageProcessor
from calsize import BoundingBoxAnalyzer
import os

# Global variable for pixels per metric
pixelsPerMetric = None

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
            return removedBG  # Return the loaded image
        else:
            print("Error: Could not load the image for display.")

    except Exception as e:
        print(e)
        
# Function to track fingers for isolation
def trackFinger(image):
    processor = HandImageProcessor()
    processed_image = processor.process_hand_image(image)  # Ensure this returns a NumPy array
    return processed_image  # Return the processed image directly
    
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the reference object in mm")
args = vars(ap.parse_args())

# Load the background-removed image
output_filename = os.path.splitext(args['image'])[0] + 'BG.jpg'

noBG = removeBG(args["image"], output_filename)
noBG = trackFinger(noBG)

img = noBG.copy()

#orig = cv2.imread(args['image'])

# Convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (11, 11), 0)

# Perform edge detection
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cv2.imshow('edge detection', edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

# Sort contours from top to bottom
cnts,_ = contours.sort_contours(cnts, method='top-to-bottom')

# Check if any contours were found
if len(cnts) == 0:
    print("No contours found.")
else:
    
    calSize = BoundingBoxAnalyzer(img, args["width"])
    
    # initialize the reference object as the first contour on sorted list
    reference_cnts = cnts[0]  
    
    # Measure the reference object
    calSize.cal_reference_size(reference_cnts)

    cv2.imshow("Reference Object", img)

    # All other contours, assumed to include the hand
    # Identify hand and fingers, skipping the reference object
    hand_cnts = cnts[1:]
    finger_cnts, _ = contours.sort_contours(hand_cnts, method="top-to-bottom")

    # Filter out any small or irrelevant contours
    min_area = 1000  # Adjust based on expected finger contour size
    finger_cnts = [c for c in finger_cnts if cv2.contourArea(c) > min_area]
    #put number in sorted contours
    # Draw contours and label them
    for (i, c) in enumerate(finger_cnts, start=0):
        # Compute the center of the contour and draw it
        M = cv2.moments(c)
        if M["m00"] != 0:  # Avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            calSize.cal_finger_size(c)  # Calculate size if needed
            
            # Draw the contour and label it
            #cv2.drawContours(img_with_ref_obj, [c], -1, (0, 255, 0), 2)
            #cv2.putText(img_with_ref_obj, f"{i + 1}", (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the final image with contours and labels
    cv2.imshow(f"Calculated Size for " + output_filename, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove(output_filename)