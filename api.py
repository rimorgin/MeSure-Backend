
from typing import Tuple
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from flask import Flask, request, jsonify

from PIL import Image
import shadowremover


app = Flask(__name__)

def remove_shadow(
                    image,
                    save: bool = False,
                    ab_threshold: int = 256,
                    lab_adjustment: bool = False,
                    region_adjustment_kernel_size: int = 10,
                    shadow_dilation_kernel_size: int = 5,
                    shadow_dilation_iteration: int = 3,
                    shadow_size_threshold: int = 2500,
                    verbose: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    

    shadow_clear, mask = shadowremover.remove_shadows(image,
                                        ab_threshold,
                                        lab_adjustment,
                                        region_adjustment_kernel_size,
                                        shadow_dilation_iteration,
                                        shadow_dilation_kernel_size,
                                        shadow_size_threshold,
                                        verbose=verbose)
    if save:
        cv2.imwrite('shadow_clear.jpg', shadow_clear)
        cv2.imwrite('mask.jpg', mask)
        
    return shadow_clear

def scale_image(image, max_width=700, max_height=700):
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

# Function to show array of images (intermediate results)
def show_images(images, filename: str = None):
    for i, img in enumerate(images):
        img = scale_image(img)
        if filename is not None:
            cv2.imshow(filename, img)
        else: cv2.imshow("image_" + str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image_cnts(image, method='top-to-bottom', show_image=False):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.Canny(image, 100, 200)
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

    # Step 5: Sort contours based on the method (default: top-to-bottom)
    (cnts, _) = contours.sort_contours(cnts, method=method)

    # Step 6: Filter out small contours (area threshold > 100)
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    print(f"Found {len(cnts)} contours")
    
    return cnts

def detect_coin(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # Reduce noise

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        gray,
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
        cv2.imwrite("Detected Coin.jpg", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return circles[0][0]  # Return the first detected circle (x, y, radius)
    else:
        return None


def detect_finger_contour(image):
    # Process the image to detect the finger contour
    cnts = process_image_cnts(image)
    return None

def calculate_scale_factor(coin_radius_px, coin_diameter_mm):
    # Calculate pixels per millimeter
    pixelsPerMetric = (2 * coin_radius_px) / coin_diameter_mm
    return pixelsPerMetric

def measure_finger_width(cnts, scale_factor):


    # Get the bounding box of the finger contour
    (x, y, w, h) = cv2.boundingRect(cnts)

    # Convert width to millimeters
    finger_width_mm = w / scale_factor
    return finger_width_mm


@app.route("/measure-finger", methods=['POST'])
def measure_finger():
    if 'image' not in request.files or 'width' not in request.form or 'finger' not in request.form:
        return jsonify({"error": "No image, coin diameter, or finger name provided"}), 400

    files = request.files.getlist('image')
    width_str = float(request.form.get('width', '').strip())
    finger_str = request.form.get('finger', '').strip().capitalize()
    
    # Validate inputs
    if len(files) != 2:
        return jsonify({"error": "Exactly 2 images must be uploaded"}), 400
    if finger_str not in {"Thumb", "Index", "Middle", "Ring", "Pinky"}:
        return jsonify({"error": "Invalid finger name"}), 400
    
    
    # reference object known diameter
    coin_diameter_mm = width_str
    finger_name = finger_str
    results = []
    
    avg_finger_width = []

    for file in files:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            results.append({"error": f"Invalid image file: {file.filename}"})
            continue
        
        shadow_clear = remove_shadow(image, save=True)

        # Detect the coin
        coin = detect_coin(image)  # or detect_coin_contour(image)
        if coin is None:
            results.append({"error": f"No coin detected in {file.filename}"})
            continue

        # Calculate scale factor
        (x, y, radius) = coin
        scale_factor = calculate_scale_factor(radius, coin_diameter_mm)
        
        '''
        cnts = process_image_cnts(image)

        if not cnts:
            return None
        
        # Assume the largest contour is the finger
        cnts = max(cnts, key=cv2.contourArea)
        

        # Measure finger width
        finger_width_mm = measure_finger_width(cnts, scale_factor)
        if finger_width_mm is None:
            results.append({"error": f"Failed to measure finger width in {file.filename}"})
            continue
        
        avg_finger_width.append(finger_width_mm)
        

        #results.append({"filename": file.filename,"finger_width_mm": finger_width_mm})
        
    avg_finger_width = np.mean(avg_finger_width)
    
        
    results.append({'finger': finger_name, 'width': avg_finger_width})
    '''
    results.append({'coin': scale_factor})

    return jsonify(results), 200

@app.route("/healthz", methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    #app.run()
    app.run(host='0.0.0.0', port=8080, debug=True)

