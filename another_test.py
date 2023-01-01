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
gray = cv2.GaussianBlur(gray, (11, 11), 0)

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
    c = max(cnts, key = lambda x: cv2.contourArea(x))

    hull = cv2.convexHull(c)
    hull_indices = cv2.convexHull(c, returnPoints=False)
    
    #approx the contour a little
    epsilon = 0.0005 * cv2.arcLength (c,True)
    approx= cv2.approxPolyDP(c,epsilon,True)
    
    #make convex hull around hand
    hull = cv2.convexHull(c)
    
    #define area of hull and area of hand
    areahull = cv2.contourArea(c)
    areacnt = cv2.contourArea(c)
    
    #find the percentage of area not covered by hand in convex hull
    arearatio=((areahull-areacnt)/areacnt)*100
    
    # Draw the original contour and the convex hull on the image
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)  # Original contour in green
    cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)  # Convex hull in blue
    
    cv2.imshow('Convex Hull', img)
    
    hull = cv2.convexHull(approx, returnPoints=False)
    hull[::-1].sort(axis=0)
    defects = cv2.convexityDefects(approx, hull)
    
    # Check if convexity defects are present
    if defects is not None:
        defects = defects[:, 0, :]  # Reshape defects
        
        # Sort defects by depth (distance between the farthest point and the convex hull)
        sorted_defects = defects[np.argsort(defects[:, 3]), :]
        
        # Keep only the largest defects (likely corresponding to finger segments)
        # Here, we are assuming that the top 5 largest defects correspond to fingers
        finger_defects = sorted_defects[-5:]  # Get 5 largest defects
        
        start_points, end_points, far_points = [], [], []
        
        for i in range(len(finger_defects)):
            s, e, f, d = finger_defects[i]
            start_points.append(tuple(approx[s][0]))  # Start of the defect
            end_points.append(tuple(approx[e][0]))    # End of the defect
            far_points.append(tuple(approx[f][0]))    # Farthest point of the defect

            # Draw the defect
            #cv2.line(img, start_points[-1], end_points[-1], (0, 255, 0), 2)  # Line between start and end
            cv2.circle(img, far_points[-1], 5, (0, 0, 255), -1)              # Defect point

        # Convert lists to arrays for easier manipulation
        far_points = np.array(far_points)

        # Step 1: Sort by the Y-coordinate (to find the top-most points first)
        sorted_indices = np.argsort(far_points[:, 1])
        far_points = far_points[sorted_indices]
        start_points = np.array(start_points)[sorted_indices]
        end_points = np.array(end_points)[sorted_indices]

        # Step 2: Calculate a threshold to eliminate wrist points
        average_finger_height = np.mean(far_points[:, 1])
        threshold = average_finger_height + (0.2 * (far_points[:, 1].max() - far_points[:, 1].min()))

        # Filter points above the threshold (likely fingers, not the wrist)
        valid_indices = np.where(far_points[:, 1] < threshold)[0]
        
        # Use only the valid points
        far_points = far_points[valid_indices]
        start_points = start_points[valid_indices]
        end_points = end_points[valid_indices]

        # Draw filtered points
        for pt in far_points:
            cv2.circle(img, tuple(pt), 5, (255, 0, 0), -1)  # Highlight valid points (in blue)

        # Show the final image with convexity defects
        cv2.imshow("Convexity Defects", img)
    else:
        print("No convexity defects found.")



    # Show the final image with the original contour, convex hull, and convexity defects
    

cv2.waitKey(0)
cv2.destroyAllWindows()
os.remove(output_filename)
