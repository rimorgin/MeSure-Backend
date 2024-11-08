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

    def finger_tracking(self, image):
        # Process the hand image passed as an object.
        
        image = cv2.flip(image, 1)
        h, w, c = image.shape  # Get the height, width, and number of channels of the image

        # Create a blank mask with the same dimensions as the image
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert the image to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        hand_label = None  # Initialize hand_label
        
        # Process each detected hand in the image
        if results.multi_hand_landmarks:
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                
                hand_label = handType.classification[0].label  # Get hand type (Left or Right)
                
                # Define offset adjustments based on handedness
                if hand_label == 'Left':
                    adjustments = [-20, 20, 25, 0]  # Adjustments for left hand
                elif hand_label == 'Right':
                    adjustments = [20, 20, -25, 0]  # Adjustments for right hand

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
                pip_points = []

                # Process each finger
                for i, (finger_name, (mcp, pip)) in enumerate(finger_joints.items()):
                    # Calculate midpoint and distance between MCP and PIP
                    midpoint, distance = self.calculate_midpoint_and_distance(mcp[0], mcp[1], pip[0], pip[1])

                    # Adjust midpoints based on handedness
                    if finger_name == "Thumb":
                        midpoint = (midpoint[0] + adjustments[0], midpoint[1] + adjustments[1])  # Adjust left or right
                    elif finger_name == "Pinky":
                        midpoint = (midpoint[0] + adjustments[2], midpoint[1] + adjustments[3])  # Adjust left or right

                    # Store the adjusted midpoint for drawing lines later
                    midpoints.append(midpoint)
                    
                    # Adjust pip_points based on handedness
                    if finger_name == "Thumb":
                        pip = (pip[0] + adjustments[0], pip[1] + adjustments[1])  # Adjust left or right
                    elif finger_name == "Pinky":
                        pip = (pip[0] + adjustments[2], pip[1] + adjustments[3])  # Adjust left or right
                        
                    # Store the PIP points for drawing lines later
                    pip_points.append(pip)

                # Create a list of points for filling
                fill_points = []

                # Combine midpoints and pip points for filling
                for i in range(len(midpoints)):
                    fill_points.append(midpoints[i])
                for i in range(len(pip_points) - 1, -1, -1):  # Reverse order for the bottom line
                    fill_points.append(pip_points[i])

                # Convert to numpy array for fillPoly
                fill_points = np.array(fill_points, np .int32)

                # Fill the polygon area
                cv2.fillPoly(mask, [fill_points], (255, 255, 255))  # Fill with white color
                
                
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(image, mask)
        
        # Flip the mask along the vertical axis
        result = cv2.flip(result, 1)

        return result, hand_label
    
    def wrist_tracking(self, image):
        # Process the hand image passed as an object.
        
        image = cv2.flip(image, 1)
        h, w, c = image.shape  # Get the height, width, and number of channels of the image

        # Create a blank mask with the same dimensions as the image
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert the image to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        hand_label = None  # Initialize hand_label
        
        # Process each detected hand in the image
        if results.multi_hand_landmarks:
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                
                hand_label = handType.classification[0].label  # Get hand type (Left or Right)

                print(hand_label)

                 # Get the wrist joint coordinates directly
                wrist_x = int(handLms.landmark[0].x * w)
                wrist_y = int(handLms.landmark[0].y * h)
                 # Coordinates for the wrist joint
                
                
                # Adjust wrist joint based on handedness
                if hand_label == 'Left':
                    # Define points for left handedness
                    wrist_points = [
                        (wrist_x + 50, wrist_y),
                        (wrist_x + 50, wrist_y + 150),
                        (wrist_x + 50, wrist_y - 150),
                        (wrist_x + 125, wrist_y),
                        (wrist_x + 125, wrist_y + 150),
                        (wrist_x + 125, wrist_y - 150)
                    ]
                elif hand_label == 'Right':
                    # Define points for Right handedness
                    wrist_points = [
                        (wrist_x - 50, wrist_y),
                        (wrist_x - 50, wrist_y + 150),
                        (wrist_x - 50, wrist_y - 150),
                        (wrist_x - 125, wrist_y),
                        (wrist_x - 125, wrist_y + 150),
                        (wrist_x - 125, wrist_y - 150)
                    ]

                # Draw lines connecting the wrist points
                cv2.line(image, wrist_points[0], wrist_points[1], (0, 0, 0), 2)  # 1st vertical line
                cv2.line(image, wrist_points[0], wrist_points[2], (0, 0, 0), 2)  # Extend 1st line up
                cv2.line(image, wrist_points[3], wrist_points[4], (0, 0, 0), 2)  # 2nd vertical line
                cv2.line(image, wrist_points[3], wrist_points[5], (0, 0, 0), 2)  # Extend 2nd line up

                # Define polygon points to fill the area between the lines
                polygon_points = np.array([
                    wrist_points[0], wrist_points[1], wrist_points[4], wrist_points[3],
                    wrist_points[5], wrist_points[2]
                ])
                
                # Fill the polygon on the mask
                cv2.fillPoly(mask, [polygon_points], (255, 255, 255))  # Fill with white color
                

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, mask)
        
        # Flip the mask along the vertical axis
        result = cv2.flip(result, 1)

        #return result, hand_label
        
        return result, hand_label



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
