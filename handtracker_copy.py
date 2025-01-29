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

    def finger_tracking(self, image, finger_name):
        """Track the specific finger and return the midpoint and distance between MCP and PIP (or CMC and MCP for thumb)."""
        
        image = cv2.flip(image, 1)  # Flip the image for correct coordinates
        h, w, c = image.shape  

        # Create a blank black mask
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # Define finger joint mappings
        finger_joints = {
            "Thumb": [1, 2],  # CMC to MCP
            "Index": [5, 6],  # MCP to PIP
            "Middle": [9, 10],  # MCP to PIP
            "Ring": [13, 14],  # MCP to PIP
            "Pinky": [17, 18]  # MCP to PIP
        }
        
        # Validate input
        if finger_name not in finger_joints:
            print(f"Invalid finger name: {finger_name}")
            return image, None  # Return image and None values if invalid
        
        midpoint = None
        
        # Process detected hands
        if results.multi_hand_landmarks:
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                hand_label = handType.classification[0].label  

                # Extract landmarks list and adjust for image size
                mylmList = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(handLms.landmark)]
                
                # Get the specific finger's keypoints for MCP and PIP (or CMC for Thumb)
                mcp_x, mcp_y = mylmList[finger_joints[finger_name][0]][1:]  # MCP joint
                pip_x, pip_y = mylmList[finger_joints[finger_name][1]][1:]  # PIP joint
                
                # Calculate the midpoint between the two joints
                midpoint, distance = self.calculate_midpoint_and_distance(mcp_x, mcp_y, pip_x, pip_y)
                '''
                # Draw the joints and midpoint (for visualization purposes)
                cv2.circle(mask, (mcp_x, mcp_y), 5, (255, 0, 0), -1)  # Red for MCP
                cv2.circle(mask, (pip_x, pip_y), 5, (0, 255, 0), -1)  # Green for PIP
                cv2.circle(mask, midpoint, 5, (0, 0, 255), -1)  # Blue for midpoint
                '''
                
                cv2.line(mask, (midpoint[0], midpoint[1]), (pip_x, pip_y), (255, 255, 255), 150) 

        # Apply mask properly
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        result = cv2.bitwise_and(image, image, mask=mask_gray)
        result = cv2.flip(result, 1)

        return result, hand_label
