import cv2
import math
import mediapipe as mp
import numpy as np

class HandImageProcessor:
    def __init__(self):
        # Initialize Mediapipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        """Calculate distance between two points in 2D space."""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def calculate_midpoint_and_distance(x1, y1, x2, y2):
        """Calculate the midpoint and distance between two points."""
        midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
        distance = HandImageProcessor.calculate_distance(x1, y1, x2, y2)
        return midpoint, distance

    def process_hand_image(self, image):
        """Process the hand image passed as an object."""
        
        #image = cv2.flip(image,1)
        h, w, c = image.shape  # Get the height, width, and number of channels of the image

        # Convert the image to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # Process each detected hand in the image
        if results.multi_hand_landmarks:
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                
                hand_label = handType.classification[0].label  # Get hand type (Left or Right)
                
                print(hand_label)
                
                #if hand_label == 'Left':
                #    pass
                #elif hand_label == 'Right':
                #   image = cv2.flip(image, 1)
                
                mylmList = []

                # Extract landmark points and store them in lists
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    mylmList.append([id, px, py])

                # Coordinates for each finger's MCP and PIP joints
                finger_joints = {
                    "Thumb": (mylmList[2][1:3], mylmList[3][1:3]),
                    "Index": (mylmList[5][1:3], mylmList[6][1:3]),
                    "Middle": (mylmList[9][1:3], mylmList[10][1:3]),
                    "Ring": (mylmList[13][1:3], mylmList[14][1:3]),
                    "Pinky": (mylmList[17][1:3], mylmList[18][1:3])
                }

                midpoints = []  # List to store midpoints

                #left_handed = [20,20,25,0]
                #right_handed = []
                # Process each finger
                for finger_name, (mcp, pip) in finger_joints.items():
                    # Calculate midpoint and distance between MCP and PIP
                    midpoint, distance = self.calculate_midpoint_and_distance(mcp[0], mcp[1], pip[0], pip[1])

                   
                    # Adjust the midpoints for the thumb and pinky
                    if finger_name == "Thumb":
                        midpoint = (midpoint[0] - 20, midpoint[1] + 20)  # Adjust left
                    elif finger_name == "Pinky":
                        midpoint = (midpoint[0] + 25, midpoint[1])  # Adjust right
              
                        
                    # Store the adjusted midpoint for drawing lines later
                    midpoints.append(midpoint)

                # Draw lines connecting all midpoints without gaps
                for i in range(len(midpoints) - 1):
                    start_point = midpoints[i]
                    end_point = midpoints[i + 1]
                    cv2.line(image, start_point, end_point, (0, 0, 0), 3)  # Black line connecting midpoints

        return image


"""
# Example usage:
image_path = 'sample2.jpg'
image = cv2.imread(image_path)

processor = HandImageProcessor()
processed_image = processor.process_hand_image(image)

cv2.imshow("Processed Image", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
