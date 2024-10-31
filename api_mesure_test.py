api_mesure_test

from flask import Flask, request, jsonify, send_file
from scipy.spatial import distance as dist
from imutils import perspective, contours
import numpy as np
import imutils
import cv2
import os

# Import custom modules
from bgremover import BackgroundRemover
from fingertracker import HandImageProcessor
from addreference import ImageOverlay

app = Flask(_name_)

# Global variable for pixels per metric, used for measurement
pixelsPerMetric = None

def midpoint(ptA, ptB):
    """Calculate midpoint between two points."""
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def calSize(contour, img_with_ref_obj, pixelsPerMetric):
    """Calculate and annotate the size of a given contour on the image."""
    # Compute the rotated bounding box of the contour
    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box) if imutils.is_cv4() else cv2.cv.BoxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv2.drawContours(img_with_ref_obj, [box.astype("int")], -1, (255, 255, 255), 2)

    # Get the midpoints between box corners for dimension calculation
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Draw measurement lines on image
    cv2.line(img_with_ref_obj, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(img_with_ref_obj, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    # Calculate dimensions
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # Annotate dimensions on the image
    cv2.putText(img_with_ref_obj, "{:.1f}".format(dimA), (int(tltrX - 15), int(tltrY + 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(img_with_ref_obj, "{:.1f}".format(dimB), (int(trbrX + 10), int(trbrY)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

def remove_background(image_path, output_path):
    """Remove the background from an image."""
    remover = BackgroundRemover(image_path)
    try:
        output_image = remover.process_image()
        output_image = output_image.convert('RGB') if output_image.mode == 'RGBA' else output_image
        output_image.save(output_path)
        return cv2.imread(output_path)
    except Exception as e:
        print(f"Background removal failed: {e}")
        return None

def process_image(image_path, ref_width):
    """Main image processing function to remove background, overlay reference, and measure dimensions."""
    global pixelsPerMetric
    pixelsPerMetric = None  # Reset pixels per metric

    # Remove background and load image
    output_path = os.path.splitext(image_path)[0] + '_BG.jpg'
    img = remove_background(image_path, output_path)
    if img is None:
        return {"error": "Error removing background."}, 500

    # Overlay reference object
    reference_image = cv2.imread('NGCCoins.png', cv2.IMREAD_UNCHANGED)
    processor = HandImageProcessor()
    processed_hand_image = processor.process_hand_image(output_path)

    overlay = ImageOverlay(processed_hand_image, reference_image, overlay_size_mm=23.0, ppi=96)
    img_with_ref_obj = overlay.add_overlay((10, 10))

    # Preprocess the image for contour detection
    gray = cv2.cvtColor(img_with_ref_obj, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts, _ = contours.sort_contours(cnts, method='top-to-bottom')

    if not cnts:
        return {"error": "No contours found."}, 404

    # Set pixelsPerMetric using reference object
    reference_contour = cnts[0]
    box = cv2.minAreaRect(reference_contour)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    dB = dist.euclidean(midpoint(box[0], box[3]), midpoint(box[1], box[2]))
    pixelsPerMetric = dB / ref_width

    # Process finger contours (limit to 5)
    finger_contours = [c for c in cnts[1:] if cv2.contourArea(c) > 1000]
    for contour in finger_contours[:5]:
        calSize(contour, img_with_ref_obj, pixelsPerMetric)

    # Save and return processed image
    processed_image_path = "processed_image.jpg"
    cv2.imwrite(processed_image_path, img_with_ref_obj)
    os.remove(output_path)  # Clean up background removed image

    return processed_image_path

@app.route('/measure_finger', methods=['POST'])
def measure_finger():
    """Endpoint to measure finger dimensions in uploaded image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    width = float(request.form['width'])  # Retrieve reference object width

    # Save image temporarily
    temp_path = os.path.join('/tmp', file.filename)
    file.save(temp_path)

    # Process image
    processed_image_path = process_image(temp_path, width)
    if isinstance(processed_image_path, dict):
        return jsonify(processed_image_path), processed_image_path[1]  # Error response

    # Send processed image as response
    response = send_file(processed_image_path, mimetype='image/jpeg')

    # Clean up temporary files
    os.remove(temp_path)
    os.remove(processed_image_path)
    return response

if _name_ == '_main_':
    app.run(debug=True)