from imutils import contours
import imutils
import cv2
import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from bgremover import BackgroundRemover
from fingertracker import HandImageProcessor
from calsize import BoundingBoxAnalyzer
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def removeBG(image, filepath):
    remover = BackgroundRemover(image)
    try:
        output_image = remover.process_image()
        if output_image.mode == 'RGBA':
            output_image = output_image.convert('RGB')
        output_image.save(filepath)
        removedBG = cv2.imread(filepath)
        return removedBG if removedBG is not None else None
    except Exception as e:
        print(f"Background removal error: {e}")
        return None

def trackFinger(image):
    processor = HandImageProcessor()
    processed_image = processor.process_hand_image(image)
    return processed_image

def process_image_cnts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Check if any contours were found
    if len(cnts) == 0:
        print("No contours found")
        return []  # Return an empty list instead of proceeding
    
    cnts, _ = contours.sort_contours(cnts, method='top-to-bottom')
    return cnts

@app.route('/measure', methods=['POST'])
def measure():
    if 'image' not in request.files or 'width' not in request.form:
        return jsonify({"error": "No image file or width provided"}), 400

    file = request.files['image']
    reference_width = float(request.form['width'])

    # Convert the uploaded image file to an OpenCV image
    orig_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if orig_image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Process the original image to find reference object contours
    reference_cnts = process_image_cnts(orig_image)
    if not reference_cnts:
        return jsonify({"error": "No contours found in the image"}), 400

    calSize = BoundingBoxAnalyzer(orig_image, reference_width)
    reference_cnt = reference_cnts[0]
    calSize.cal_reference_size(reference_cnt)

    # Load the background-removed image
    output_filename = "temp_BG_removed.jpg"
    noBG = removeBG(orig_image, output_filename)
    if noBG is None:
        return jsonify({"error": "Failed to remove background"}), 500

    noBG = trackFinger(noBG)

    # Process the background-removed image to find finger contours
    finger_cnts = process_image_cnts(noBG)
    if not finger_cnts:
        return jsonify({"error": "No finger contours found"}), 400

    min_area = 1000
    finger_cnts = [c for c in finger_cnts if cv2.contourArea(c) > min_area]

    finger_measurements = []
    for (i, c) in enumerate(finger_cnts):
        M = cv2.moments(c)
        if M["m00"] != 0:
            finger_data = calSize.cal_finger_size(c)
            finger_measurements.append(finger_data)

    os.remove(output_filename)

    return jsonify({"finger_measurements": finger_measurements})

if __name__ == '__main__':
    app.run(debug=True)
