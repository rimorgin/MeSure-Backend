from imutils import contours
import imutils
import cv2
import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from bgremover import BackgroundRemover
from handtracker import HandImageProcessor
from calsize import BoundingBoxAnalyzer
from io import BytesIO
from PIL import Image

app = Flask(__name__)
def detect_objects_on_white(image):
    """
    Detect objects (e.g., hand and coin) placed on a white background.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use a binary inverse threshold to detect non-white regions
    _, mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

    # Use morphological operations to refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply the mask to the original image to isolate non-white regions
    result_image = cv2.bitwise_and(image, image, mask=refined_mask)

    return result_image, refined_mask

def removeBG(image, filepath):
    # Convert the OpenCV image (NumPy array) to a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    remover = BackgroundRemover(pil_image)  # Pass the PIL Image to BackgroundRemover
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
    processed_image, hand_label = processor.finger_tracking(image)
    return processed_image, hand_label

def trackWrist(image):
    processor = HandImageProcessor()
    processed_image, hand_label = processor.wrist_tracking(image)
    return processed_image, hand_label

def scale_down_image(image, max_width=800, max_height=600):
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


def process_image_cnts(image, method):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(gray, threshold1=50, threshold2=100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Check if any contours were found
    if len(cnts) == 0:
        print("No contours found")
        return []  # Return an empty list instead of proceeding
    
    cnts, _ = contours.sort_contours(cnts, method=method)
    return cnts

@app.route('/measure-fingers', methods=['POST'])
def measure():
    if 'image' not in request.files or 'width' not in request.form:
        return jsonify({"error": "No image file or width provided"}), 400

    file = request.files['image']
    reference_width = float(request.form['width'])
    
    file_size = round(len(file.read()) / 1024, 1)  # size in KB
    if file_size == 0:
        return jsonify({"error": "Uploaded image is empty (0 bytes)"}), 400

    # Reset file pointer to the beginning for further processing
    file.seek(0)

    # Convert the uploaded image file to an OpenCV image
    orig_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if orig_image is None:
        return jsonify({"error": "Invalid image file"}), 400
    
    
    # Save original dimensions for later resizing
    original_height, original_width = orig_image.shape[:2]

    # Scale down image for faster processing
    scaled_image = scale_down_image(orig_image)

    # Process the scaled-down image to find reference object contours
    reference_cnts = process_image_cnts(scaled_image, 'top-to-bottom')
    if not reference_cnts:
        return jsonify({"error": "No contours found in the image"}), 400

    calSize = BoundingBoxAnalyzer(scaled_image, reference_width)
    reference_cnt = reference_cnts[0]
    calSize.cal_reference_size(reference_cnt)
    
    cv2.imwrite('reference.jpg', scaled_image)

    # Load the background-removed image
    output_filename = "temp_BG_removed.jpg"
    noBG = removeBG(scaled_image, output_filename)
    if noBG is None:
        return jsonify({"error": "Failed to remove background"}), 500

    noBG, hand_label = trackFinger(noBG)

    # Process the background-removed image to find finger contours
    finger_cnts = process_image_cnts(noBG, 'left-to-right')
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

    # Restore original image size after processing
    restored_image = cv2.resize(scaled_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    # Convert to PIL for sending back to the client
    processed_image = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)
    processed_image = Image.fromarray(processed_image)
    
    img_io = BytesIO()
    processed_image.save(img_io, format='JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/measure-wrist', methods=['POST'])
def measure_wrist():
    if 'image' not in request.files or 'width' not in request.form:
        return jsonify({"error": "No image file or reference width provided"}), 400

    file = request.files['image']
    reference_width = float(request.form['width'])

    # Check for 0-byte file
    file_size = round(len(file.read()) / 1024, 1)
    if file_size == 0:
        return jsonify({"error": "Uploaded image is empty (0 bytes)"}), 400
    file.seek(0)

    # Read image and convert to OpenCV format
    orig_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if orig_image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Save original dimensions for later resizing
    original_height, original_width = orig_image.shape[:2]

    # Scale down image for faster processing
    scaled_image = scale_down_image(orig_image)

    # Process the scaled-down image to find reference object contours
    reference_cnts = process_image_cnts(scaled_image, 'top-to-bottom')
    if not reference_cnts:
        return jsonify({"error": "No contours found in the image"}), 400

    calSize = BoundingBoxAnalyzer(scaled_image, reference_width)
    reference_cnt = reference_cnts[0]
    calSize.cal_reference_size(reference_cnt)

    # Load the background-removed image
    output_filename = "temp_BG_removed.jpg"
    noBG = removeBG(scaled_image, output_filename)
    if noBG is None:
        return jsonify({"error": "Failed to remove background"}), 500

    noBG, hand_label = trackWrist(noBG)
    
    # Process the background-removed image to find wrist contours
    wrist_cnt = process_image_cnts(noBG, 'bottom-to-top')
    if not wrist_cnt:
        return jsonify({"error": "No wrist contour found"}), 400
    
    # Get the largest contour in the image, which is the wrist
    c = max(wrist_cnt, key=cv2.contourArea)
    calSize.cal_wrist_size(c)

    # Restore original image size after processing
    restored_image = cv2.resize(scaled_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    # Convert to PIL for sending back to the client
    processed_image = cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)
    processed_image = Image.fromarray(processed_image)
    
    img_io = BytesIO()
    processed_image.save(img_io, format='JPEG')
    img_io.seek(0)

    os.remove(output_filename)

    return send_file(img_io, mimetype='image/jpeg')
    
@app.route("/", methods=['GET'])
def index():
    return ({"msg":"Connected to internet and MeSure API"})

@app.route("/healthz", methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    #app.run()
    app.run(host='0.0.0.0', port=8080, debug=True)
