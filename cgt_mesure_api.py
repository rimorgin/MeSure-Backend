"""
Filename: init.py
Usage: This script will measure different objects in the frame using a reference object of known dimension. 
The object with known dimension must be the leftmost object.
"""

from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from flask import Flask, request, jsonify
from calsize import BoundingBoxAnalyzer
from handtracker_copy import HandImageProcessor
from bgremover import BackgroundRemover
from PIL import Image
import os

app = Flask(__name__)

def trackFinger(image, finger_name, filename='mask'):
    processor = HandImageProcessor()
    processed_image, hand_label = processor.finger_tracking(image, finger_name=finger_name, filename=filename)
    return processed_image, hand_label

def trackWrist(image):
    processor = HandImageProcessor()
    processed_image, hand_label = processor.wrist_tracking(image)
    return processed_image, hand_label

def removeBG(image, filepath):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
    remover = BackgroundRemover(image)

    try:
        output_image = remover.process_image()
        if output_image is None:
            print("Background removal returned None.")
            return None
        if isinstance(output_image, np.ndarray):
            output_image = Image.fromarray(output_image)
        if output_image.mode == 'RGBA':
            output_image = output_image.convert('RGB')
    
        output_image.save(filepath)
        removedBG = cv2.imread(filepath)
        if removedBG is None:
            print("Failed to read the saved image.")
            return None
        
        return removedBG
    except Exception as e:
        print(f"Background removal error: {e}")
        return None

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


@app.route("/measure-finger", methods=['POST'])
def measure_finger():
    # Check if the POST request has the file part
    
    if 'image' not in request.files or 'width' not in request.form or 'finger' not in request.form:
        return jsonify({"error": "No image files or reference width or finger name provided"}), 400

    files = request.files.getlist('image')  # Corrected field name
    width_str = request.form.get('width', '').strip()
    finger_str = request.form.get('finger', '').strip().capitalize()
    
    # Define finger joint mappings
    allowed_fingers = {
        "Thumb",
        "Index",
        "Middle",
        "Ring",
        "Pinky",
    }
    
    # Validate input
    if len(files)is None or len(files) == 1:
        return jsonify({"error": "Total uploaded image should be 2"}), 400
    elif width_str == '':
        return jsonify({"error": "Reference width not provided"}), 400
    elif not width_str.isdigit():
        return jsonify({"error": "Invalid format and must be numeric"}), 400
    elif finger_str == '':
        return jsonify({"error": "Finger name not provided"}), 400
    elif finger_str not in allowed_fingers:
        return jsonify({"error": f"Invalid finger name not in {allowed_fingers}"}), 400
    

    reference_width = float(width_str)
    finger_name = finger_str
    
    final_avg_measurement = []
    initial_avg_measurement = []
    #response or results
    results = []

    successful_iteration = 0
    iteration_hand_label = ''
    
    
    # process each uploaded image
    for file in files:

        # Read image and convert to OpenCV format
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            results.append({"error": f"Invalid image file of unknown format on uploaded image {successful_iteration+1}"}), 400
            break
            
        image = scale_image(image)
        # add image sharpening
        kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32) 
        image = cv2.filter2D(image, -1, kernel)
        
        # initialize bounding box analyzer
        calSize = BoundingBoxAnalyzer(image, reference_width)
        # Process the image
        try:
            cnts = process_image_cnts(image)
            
            # Check if contours were found, early exit if NONE
            if cnts is None:
                jsonify({"error": f"Failed to process the {successful_iteration+1} image"}), 400
                break
            
            reference_cnt = cnts[0]
            
            # Get the reference pixel size
            calSize.cal_reference_size(reference_cnt)
            
            cv2.imwrite(f'processed/reference-{file.filename}.png', image)
            
            # Load the background-removed image
            output_filename = f"processed/{file.filename}-BG-removed.jpg"
            
            noBG = removeBG(image, output_filename)
            if noBG is None:
                results.append({"error": "Failed to remove background"})
                break
                
            show_images([noBG], filename=f'noBG-{file.filename}')

            finger_mask, hand_label = trackFinger(noBG, finger_name, filename=file.filename)
            iteration_hand_label = hand_label
            
            if successful_iteration != 0:
                if iteration_hand_label != hand_label:
                    results.append({"error": f"Hand label mismatch on uploaded image {successful_iteration+1}"}), 400
                    break

            show_images([finger_mask], filename=f'finger-{file.filename}')
            
            # initialize finger measurements variable
            finger_measurement = []
            
             # Process the finger tracked image to find finger contours
            finger_cnts = process_image_cnts(finger_mask, show_image=True)
            if len(finger_cnts) == 0:
                results.append({"error": f"No contours found on the finger area {successful_iteration+1} image"}), 400
                break
            
            min_area = 1000
            finger_cnts = [c for c in finger_cnts if cv2.contourArea(c) > min_area]

            
            for _ in range(20):  # Perform 20 iterations
                try:
                    # Calculate measurements for this iteration
                    finger_estimated_size = [
                        calSize.cal_finger_size(c) for c in finger_cnts if cv2.moments(c)["m00"] != 0
                    ]

                    # Add this iteration's measurements to the list
                    finger_measurement.append(finger_estimated_size)
                except Exception as e:
                    return results.append({"error": f"Iteration failed: {str(e)}"}), 400
                
            # Compute the average measurements
            if len(finger_measurement) > 0:
                avg = np.mean(finger_measurement, axis=0)[0]
                
            else:
                avg = []

            initial_avg_measurement.append(avg)
            
            successful_iteration += 1
            cv2.imwrite(f'processed/final-{file.filename}.png', image)
        except Exception as e:
            results.append({"error": f"Failed to process image: {str(e)}"}), 400
            break
        
    #results.append({"message": "Objects measured successfully", "initial_avg_measurements": initial_avg_measurement})
    if len(files) == successful_iteration and successful_iteration > 0:
        final_avg_measurement = np.mean(initial_avg_measurement, axis=0)
    
        results.append({"success": "Objects measured successfully", "final_avg_measurements": final_avg_measurement}), 200

    print(results)
    # Return results for all files
    if 'success' in results:
        return jsonify(results), 200
    else:
        return jsonify(results), 400
        

@app.route("/healthz", methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    #app.run()
    app.run(host='0.0.0.0', port=8080, debug=True)