import cv2
import numpy as np
import imutils
from imutils import perspective
from imutils import contours
from handtracker import HandImageProcessor
from bgremover import BackgroundRemover
from PIL import Image


def detect_coin(image):
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)
        
        # Show the image with detected circles
        cv2.imshow("Detected Coin.jpg", image)
       
        return circles[0][0]  # Return the first detected circle (x, y, radius)
    else:
        return None

def trackFinger(image):
    processor = HandImageProcessor()
    #processed_image, hand_label = processor.finger_tracking(image, finger_name=finger_name, filename=filename)
    #return processed_image, hand_label
    hand_label, palm_orientation = processor.orientation_tracking(image)
    fingers = processor.finger_tracking(image, hand_label, palm_orientation)
    
    return fingers

def removeBG(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Input image must be a valid numpy array.")
    remover = BackgroundRemover(image)

    try:
        output_image = remover.process_image()
        if output_image is None:
            print("Background removal returned None.")
            return None
        if isinstance(output_image, Image.Image):
            output_image = np.array(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        return output_image
    except Exception as e:
        print(f"Background removal error: {e}")
        return None

def process_image_cnts(image, method='top-to-bottom', show_image=False, min_contour_area=100):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    edged = cv2.Canny(image, 50, 101, apertureSize=3, L2gradient=True)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    if show_image:
        cv2.imshow('Blurred', blur)
        cv2.imshow('Edged', edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Step 4: Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Filter out tiny contours
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > min_contour_area]

    # Step 5: Sort contours based on the method (default: top-to-bottom)
    (cnts, _) = contours.sort_contours(cnts, method=method)

    print(f"Found {len(cnts)} contours")
    
    return cnts

image = cv2.imread('images/palmup-left.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (15, 15), 0)

(x, y, radius) = detect_coin(blur)

if (x, y, radius) is not None:
    # Calculate the pixel per metric
    known_width = 23.0  # Known width of the coin in mm
    pixel_per_metric = radius * 2 / known_width

    rect = (x - radius, y - radius, 2 * radius, 2 * radius)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
    cv2.imshow("Detected Coin with Bounding Box", image)
    
    print(f"Pixel per metric: {pixel_per_metric}")

noBG = removeBG(image)
#cv2.imshow('noBG', noBG)
#cv2.waitKey(0)

fingers = trackFinger(noBG)
cv2.imshow('finger_mask', fingers)
cv2.waitKey(0)

finger_cnts = process_image_cnts(fingers)
draw = cv2.drawContours(image, finger_cnts, -1, (255,255,0), 2)
cv2.imshow('cnts', image)
cv2.waitKey(0)

for cnt in finger_cnts:
    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(box) if not imutils.is_cv3() else cv2.cv.BoxPoints(box)
    box = np.int0(box) 

    # order the points in the contour
    box = perspective.order_points(box)
    cv2.drawContours(image, [box.astype("int")], -1, (255, 255, 255), 2)

# Assuming finger_cnt is your single contour
contour_perimeter = cv2.arcLength(finger_cnts[0], True)  # Perimeter in pixels
contour_area = cv2.contourArea(finger_cnts[0])  # Area in pixels²

# Convert to real-world units
real_perimeter = contour_perimeter / pixel_per_metric
real_area = contour_area / (pixel_per_metric ** 2)

print('Perimeter:', real_perimeter, 'mm')
print('Area:', real_area, 'mm²')

cv2.imshow('rects', image)

#print(f"Handedness: {hand_label} \nPalm orientation: {palm_orientation}")


cv2.waitKey(0)
cv2.destroyAllWindows()