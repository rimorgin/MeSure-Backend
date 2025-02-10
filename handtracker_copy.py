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

    def finger_tracking(self, image, finger_name, filename):
        """Track the specific finger and return the midpoint and distance between MCP and PIP (or CMC and MCP for thumb)."""
        
        image = cv2.flip(image, 1)  # Flip the image for correct coordinates
        h, w, c = image.shape  

        # Create a blank black mask
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        
        # Define finger joint mappings
        allowed_fingers = {
            "Thumb",
            "Index",
            "Middle",
            "Ring",
            "Pinky",
        }
        
        # Validate input
        if finger_name not in allowed_fingers:
            print(f"Invalid finger name: {finger_name}")
            return image, None  # Return image and None values if invalid
    
        
        target_finger = finger_name
        
        # Process detected hands
        if results.multi_hand_landmarks:
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                hand_label = handType.classification[0].label  

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

                # Dictionary defining adjacent fingers
                adjacent_points = {
                    "Thumb": ["Index"],
                    "Index": ["Thumb", "Middle"],
                    "Middle": ["Index", "Ring"],
                    "Ring": ["Middle", "Pinky"],
                    "Pinky": ["Ring"]
                }
                print('target finger is: ', target_finger)
                # Get the fingers to process (target finger + its adjacent fingers)
                fingers_to_process = [target_finger] + adjacent_points.get(target_finger, [])
                filtered_finger_joints = {k: v for k, v in finger_joints.items() if k in fingers_to_process}

                midpoints = []  # List to store midpoints
                pip_points = []

               # Iterate over the target fingers
                for finger_name, (mcp, pip) in filtered_finger_joints.items():
                    # Get adjacent fingers from the dictionary
                    adjacent_fingers = adjacent_points.get(finger_name, [])

                    for adjacent_finger in adjacent_fingers:
                        # Check if the adjacent finger is in the filtered list
                        if adjacent_finger in filtered_finger_joints:
                            adj_mcp, adj_pip = filtered_finger_joints[adjacent_finger]
                            
                            target_midpoint, _ = self.calculate_midpoint_and_distance(mcp[0], mcp[1], pip[0], pip[1])
                            adj_target_midpoint, _ = self.calculate_midpoint_and_distance(adj_mcp[0], adj_mcp[1], adj_pip[0], adj_pip[1])

                            # Calculate midpoints between MCPs of target and adjacent finger
                            mcp_midpoint, _ = self.calculate_midpoint_and_distance(target_midpoint[0], target_midpoint[1], adj_target_midpoint[0], adj_target_midpoint[1])

                            # Calculate midpoints between PIPs of target and adjacent finger
                            pip_midpoint, _ = self.calculate_midpoint_and_distance(pip[0], pip[1], adj_pip[0], adj_pip[1])

                            # Store midpoints for further processing (drawing lines, filling polygons, etc.)
                            midpoints.append(mcp_midpoint)
                            pip_points.append(pip_midpoint)

                    # Create a list of points for filling
                    fill_points = midpoints + pip_points[::-1]  # Combine midpoints and reverse pip points

                    # Convert to numpy array for fillPoly
                    fill_points = np.array(fill_points, np.int32)

                    # Fill the polygon area
                    cv2.fillPoly(mask, [fill_points], (255, 255, 255))  # Fill with white color
                

        # Apply mask properly
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        result = cv2.bitwise_and(image, image, mask=mask_gray)
        result = cv2.flip(result, 1)
        
        cv2.imwrite(f'processed/mask-{filename}.png', result)

        return result, hand_label
