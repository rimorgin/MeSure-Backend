import cv2
import numpy as np
from scipy.spatial import distance as dist
import imutils
from imutils import perspective

class BoundingBoxAnalyzer:
    def __init__(self, img_with_ref_obj, width):
        self.img_with_ref_obj = img_with_ref_obj
        self.width = width
        self.pixelsPerMetric = None

    @staticmethod
    def midpoint(ptA, ptB):
        return (int((ptA[0] + ptB[0]) / 2.0), int((ptA[1] + ptB[1]) / 2.0))
    
    def cal_reference_size(self, cnt):
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box) if not imutils.is_cv3() else cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour
        box = perspective.order_points(box)
        cv2.drawContours(self.img_with_ref_obj, [box.astype("int")], -1, (255, 255, 255), 2)
        
        # unpack the ordered bounding box
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = self.midpoint(tl, tr)
        (blbrX, blbrY) = self.midpoint(bl, br)
        (tlblX, tlblY) = self.midpoint(tl, bl)
        (trbrX, trbrY) = self.midpoint(tr, br)
    
        # draw the midpoints on the image
        cv2.circle(self.img_with_ref_obj, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(self.img_with_ref_obj, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(self.img_with_ref_obj, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(self.img_with_ref_obj, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
        # draw lines between the midpoints
        cv2.line(self.img_with_ref_obj, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(self.img_with_ref_obj, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized
        if self.pixelsPerMetric is None:
            self.pixelsPerMetric = dB / self.width
            
        # compute the size of the object
        dimA = dA / self.pixelsPerMetric
        dimB = dB / self.pixelsPerMetric
        
        # draw the object sizes on the image
        cv2.putText(self.img_with_ref_obj, "{:.1f}".format(dimA),
                    (int(tltrX - 15), int(tltrY + 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(self.img_with_ref_obj, "{:.1f}".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        
        return dimA, dimB  # Return the dimensions as a tuple

    def cal_finger_size(self, cnt):
        # Compute the rotated bounding box of the contour
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box) if not imutils.is_cv3() else cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour
        box = perspective.order_points(box)

        # Draw the outline of the rotated bounding box
        cv2.drawContours(self.img_with_ref_obj, [box.astype("int")], -1, (0, 255, 255), 2)

        # Unpack the ordered bounding box
        (tl, tr, br, bl) = box

        # Compute the midpoints
        (tltrX, tltrY) = self.midpoint(tl, tr)  # Top midpoint
        (blbrX, blbrY) = self.midpoint(bl, br)  # Bottom midpoint
        (tlblX, tlblY) = self.midpoint(tl, bl)  # Left midpoint
        (trbrX, trbrY) = self.midpoint(tr, br)  # Right midpoint
        
        # Calculate the new midpoints for left and right sides
        (blX, blY) = bl  # Bottom-left corner coordinates
        (brX, brY) = br  # Bottom-right corner coordinates

        # Calculate the midpoint between bottom-left (bl) and top-left (tl)
        midpoint_leftX = (blX + tlblX) / 2
        midpoint_leftY = (blY + tlblY) / 2

        # Calculate the midpoint between bottom-right (br) and top-right (tr)
        midpoint_rightX = (brX + trbrX) / 2
        midpoint_rightY = (brY + trbrY) / 2

        # Draw all midpoints on the image
        #cv2.circle(self.img_with_ref_obj, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)  # Top midpoint
        #cv2.circle(self.img_with_ref_obj, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)  # Bottom midpoint
        #cv2.circle(self.img_with_ref_obj, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)  # Left midpoint
        #cv2.circle(self.img_with_ref_obj, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)  # Right midpoint
        
        # Draw the new midpoints
        cv2.circle(self.img_with_ref_obj, (int(midpoint_leftX), int(midpoint_leftY)), 5, (255, 0, 0), -1)
        cv2.circle(self.img_with_ref_obj, (int(midpoint_rightX), int(midpoint_rightY)), 5, (255, 0, 0), -1)

        # Draw lines between the new midpoints
        cv2.line(self.img_with_ref_obj, (int(midpoint_leftX), int(midpoint_leftY)), (int(midpoint_rightX), int(midpoint_rightY)), (255, 0, 255), 2)

        # Compute the Euclidean distances between the new midpoints
        dA = dist.euclidean((midpoint_leftX, midpoint_leftY), (midpoint_rightX, midpoint_rightY))  # Width

        # If the pixels per metric has not been initialized, compute it
        if self.pixelsPerMetric is None:
            self.pixelsPerMetric = dA / self.width  

        # Compute the size of the object
        dimB = dA / self.pixelsPerMetric  # Width in metric units

        # Draw the object sizes on the image
        cv2.putText(self.img_with_ref_obj, "{:.1f}".format(dimB),
                    (int(midpoint_leftX + 5), int(midpoint_leftY-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return dimB  # Return the dimension as a single value
    
    
    def cal_wrist_size(self, cnt):
        # Compute the rotated bounding box of the contour
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box) if not imutils.is_cv3() else cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")

        # Order the points in the contour
        box = perspective.order_points(box)

        # Draw the outline of the rotated bounding box
        cv2.drawContours(self.img_with_ref_obj, [box.astype("int")], -1, (0, 255, 0), 2)

        # Unpack the ordered bounding box
        (tl, tr, br, bl) = box

        # Compute the midpoints between opposite sides
        (tltrX, tltrY) = self.midpoint(tl, tr)  # Top midpoint
        (blbrX, blbrY) = self.midpoint(bl, br)  # Bottom midpoint
        (tlblX, tlblY) = self.midpoint(tl, bl)  # Left midpoint
        (trbrX, trbrY) = self.midpoint(tr, br)  # Right midpoint

        # Draw the midpoints on the image
        cv2.circle(self.img_with_ref_obj, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(self.img_with_ref_obj, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(self.img_with_ref_obj, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(self.img_with_ref_obj, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # Draw lines between the midpoints
        cv2.line(self.img_with_ref_obj, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(self.img_with_ref_obj, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # Compute the Euclidean distances between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # Height
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # Width

        # If the pixels per metric has not been initialized, compute it
        if self.pixelsPerMetric is None:
            self.pixelsPerMetric = dB / self.width

        # Compute the dimensions of the wrist
        dimA = dA / self.pixelsPerMetric  # Height in metric units
        dimB = dB / self.pixelsPerMetric  # Width in metric units

        # Draw the dimensions on the image
        cv2.putText(self.img_with_ref_obj, "{:.1f}".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(self.img_with_ref_obj, "{:.1f}".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        return dimA, dimB  # Return the wrist dimensions as a tuple

    
# Example of how to use the class
# img_with_ref_obj = cv2.imread("path_to_image.jpg")  # Your image here
# args = {"width": 10}  # Replace with actual argument
# contour = ...  # Your detected contour here
# analyzer = BoundingBoxAnalyzer(img_with_ref_obj, args["width"])
# analyzer.cal_reference_size(contour, args)  # Calculate the reference size
# analyzer.cal_finger_size(contour)  # Calculate the finger size