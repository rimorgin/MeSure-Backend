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

def sizeCalculateFingers(noBGorScaled, finger_mask, hand_label, reference_width):
    
    #cv2.imwrite('temp_image.jpg', noBGorScaled)

    gray = cv2.cvtColor(noBGorScaled, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 2)

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
    #dist_in_mm = 20.5  # Reference object's width in mm
    dist_in_mm = reference_width
    # pixel_per_mm = diameter_in_pixel / dist_in_mm
    pixel_per_mm = dist_in_pixel / dist_in_mm
    print("pixel per mm", pixel_per_mm)

   

    gray = cv2.cvtColor(finger_mask, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # print("count:", cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)
    
    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 30]
    print(len(cnts))
    
    index = 0
    finger_labels = None

    # Draw remaining contours
    left_hand = ['pinky', 'ring', 'middle', 'index', 'thumb']
    right_hand = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    finger_labels = left_hand if hand_label == "Left" else right_hand
    
    # Initialize measurements dictionary
    finger_measurements = {}
    
    
    for cnt in cnts:
        if index >= len(finger_labels):  # Stop if index exceeds the number of finger labels
            break  # Exit the loop
        
        finger_name = finger_labels[index]
        wid_values = []
        
        for _ in range(5):
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
        
            wid = euclidean(tl, tr) / pixel_per_mm
            ht = euclidean(tr, br) / pixel_per_mm
            
            # finger processing validation, width is mistakenly being horizontal value
            # width (horizontal) should not be higher than the height (vertical)
            if wid > ht:
                wid_values.append(ht)
            else:
                wid_values.append(wid)
        
        # Compute average width
        avg_wid = sum(wid_values) / len(wid_values)
        
        # Store the final average measurement as float
        finger_measurements[finger_name] = avg_wid
        cv2.drawContours(noBGorScaled, [box.astype("int")], -1, (0, 0, 255), 1)
        # Draw the measurement on the image
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
        mid_pt_vertical = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))

        cv2.putText(noBGorScaled, "{:.2f}mm".format(avg_wid), 
                        (int(mid_pt_horizontal[0] - 20), int(mid_pt_horizontal[1] + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        index += 1

    return finger_measurements, noBGorScaled

def sizeCalculateWrist(noBGorScaled, wrist_mask, hand_label, reference_width):


    gray = cv2.cvtColor(noBGorScaled, cv2.COLOR_BGR2GRAY)
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
    #dist_in_mm = 20.5  # Reference object's width in mm
    dist_in_mm = reference_width
    # pixel_per_mm = diameter_in_pixel / dist_in_mm
    pixel_per_mm = dist_in_pixel / dist_in_mm
    print(pixel_per_mm)


    gray = cv2.cvtColor(wrist_mask, cv2.COLOR_BGR2GRAY)
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

    data = []
    index = 0
    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
  
        cv2.drawContours(noBGorScaled, [box.astype("int")], -1, (0, 0, 255), 1)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
        wid = euclidean(tl, tr) / pixel_per_mm
        ht = euclidean(tr, br) / pixel_per_mm
        print("This is the ht of the wrist [{}] {}".format(index, ht))
        # if instance of the wrist width value is mistakenly being horizontal value
        if wid > ht:
            cv2.putText(noBGorScaled, "{:.2f}mm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if(hand_label == 'Left'):
                data.append("{:.2f}".format(wid))
            elif (hand_label == 'Right'):
                data.append("{:.2f}".format(wid))
        else: 
            cv2.putText(noBGorScaled, "{:.1f}mm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if(hand_label == 'Left'):
                data.append("{:.2f}".format(ht))
            elif (hand_label == 'Right'):
                data.append("{:.2f}".format(ht))
        index += 1
    return data, noBGorScaled
    
    




