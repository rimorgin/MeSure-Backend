"""
Filename: init.py
Usage: This script will measure different objects in the frame using a reference object of known dimension. 
The object with known dimension must be the leftmost object.
"""
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def scale(image, max_width=700, max_height=700):
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate scaling factors
    width_scale = max_width / original_width
    height_scale = max_height / original_height
    
    # Use the smaller scale to maintain aspect ratio
    scale = min(width_scale, height_scale)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image using high-quality interpolation
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_image


def sizeCalculate(hand_label):
    img_path = "temp_BG_removed.jpg"

    # Read image and preprocess
    image = cv2.imread(img_path)
    image = scale(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]
    print(len(cnts))

    # Reference object dimensions
    # Here for reference I have used a 20mm x 20mm square
    ref_object = cnts[0]
    # (center, radius) = cv2.minEnclosingCircle(ref_object)
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    # diameter_in_pixel = 2 * radius
    dist_in_mm = 20.5  # Reference object's width in mm
    # pixel_per_mm = diameter_in_pixel / dist_in_mm
    pixel_per_mm = dist_in_pixel / dist_in_mm
    print(pixel_per_mm)


    img_path = "finger_mask.png" 
    # img_path = "images/temp_BG_removed.jpg"


    # Read image and preprocess
    image = cv2.imread(img_path)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)
    i = int(0)

    # Draw remaining contours
    left_hand = ['pinky', 'ring', 'middle', 'index', 'thumb']
    right_hand = ['thumb', 'index', 'middle', 'ring', 'pinky']
    data = []
    index = 0
    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        # if i == 0:
        #     print('this is zero')
        #     print('this is value of tl tr ', float(euclidean(tl, tr)/ pixel_per_mm))
        #     print('this is value of tr br', float(euclidean(tr, br)/ pixel_per_mm))
        #     print('this is value of bl br ', float(euclidean(bl, br)/ pixel_per_mm))
        #     print('this is value of tl bl', float(euclidean(tl, bl)/ pixel_per_mm))
        #     print('this is value of tl br', float(euclidean(tl, br)/ pixel_per_mm))
        #     print('this is value of tr bl', float(euclidean(tr, br)/ pixel_per_mm))
        # elif i == 4:
        #     print('this is 4')
        #     print('this is value of tl tr ', float(euclidean(tl, tr)/ pixel_per_mm))
        #     print('this is value of tr br', float(euclidean(tr, br)/ pixel_per_mm))
        #     print('this is value of bl br ', float(euclidean(bl, br)/ pixel_per_mm))
        #     print('this is value of tl bl', float(euclidean(tl, br)/ pixel_per_mm))
        #     print('this is value of tr bl', float(euclidean(tr, bl)/ pixel_per_mm))
        # elif i == 1:
        #     print('this is 1')
        #     print('this is value of tl tr ', float(euclidean(tl, tr)/ pixel_per_mm))
        #     print('this is value of tr br', float(euclidean(tr, br)/ pixel_per_mm))
        #     print('this is value of bl br ', float(euclidean(bl, br)/ pixel_per_mm))
        #     print('this is value of tl bl', float(euclidean(tl, br)/ pixel_per_mm))
        #     print('this is value of tr bl', float(euclidean(tr, bl)/ pixel_per_mm))

        cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 1)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
        wid = euclidean(tl, tr) / pixel_per_mm
        print("This is the Wid of the finger [{}] {}".format(index, wid))
        ht = euclidean(tr, br) / pixel_per_mm
        cv2.putText(image, "{:.2f}mm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # cv2.putText(image, "{:.1f}mm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if(hand_label == 'Left'):
            data.append("{:.2f}".format(wid))
        elif (hand_label == 'Right'):
            data.append("{:.2f}".format(wid))
        index += 1
    return data




