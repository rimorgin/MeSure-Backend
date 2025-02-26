import cv2
import numpy as np
from handtracker import HandImageProcessor
from bgremover import BackgroundRemover
from PIL import Image
from flask import Flask, request, jsonify
from io import BytesIO
import base64
import calsize

app = Flask(__name__)

def detect_coin_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # Calculate circularity
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Filter contours based on circularity (closer to 1 is more circular)
        if 0.8 < circularity < 1.2:  # Adjust thresholds as needed
            (x, y), radius = cv2.minEnclosingCircle(c)
            return True
    return False

def detect_coin(image):
     
    # Convert to greyscale
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (9, 9), 2) 

    (h, w) = img_gray.shape[:2]
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        img_gray, 
        cv2.HOUGH_GRADIENT, 
        dp=2, 
        minDist=10, 
        param1=150, 
        param2=60, 
        minRadius=10,
        maxRadius=40
    )
    if circles is not None: 
        circles = np.uint16(np.around(circles))
        print(f"found {len(circles)} circles")
        for c in circles[0, :]:
            print(c)
            cv2.circle(image, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(image, (c[0], c[1]), 1, (0, 0, 255), 1)
        
        #cv2.imwrite("processed/circle.png",image)
        return True

    else:
        return False
    
def trackFinger(image):
    processor = HandImageProcessor()
    #processed_image, hand_label = processor.finger_tracking(image, finger_name=finger_name, filename=filename)
    #return processed_image, hand_label
    #hand_label, palm_orientation = processor.orientation_tracking(image)
    fingers, hand_label = processor.finger_tracking(image)
    
    return fingers, hand_label

def trackWrist(image):
    processor = HandImageProcessor()
    wrist, hand_label = processor.wrist_tracking(image)
    return wrist, hand_label

def removeBG(image):

    # # Convert to float32 for more precise operations
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)

    # # Reduce contrast by adjusting L-channel intensity
    # l = cv2.addWeighted(l, 1.4, np.full(l.shape, 128, dtype=np.uint8), 0, 0)

    # # Merge back and convert to BGR
    # lab = cv2.merge((l, a, b))
    # image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # cv2.imwrite("processed/contrast_adjusted.jpg", image)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Input image must be a valid numpy array.")
    remover = BackgroundRemover(image)

    try:
        output_image = remover.process_image()
        if output_image is None:
            print("Background removal returned None.")
            return None
        if isinstance(output_image, Image.Image):
            output_image = np.array(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        

        return output_image
    
        
    except Exception as e:
        print(f"Background removal error: {e}")
        return None
    
def scale_down_image(image, max_width=700, max_height=700):
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
    
@app.route("/", methods=['GET'])
def index():
    return ({"msg":"Connected to internet and MeSure API"})

@app.route("/healthz", methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/measure-fingers', methods=['POST'])
def measure_fingers():
    if 'image' not in request.files or 'width' not in request.form:
        return jsonify({"error": "No image file or width provided"}), 400

    file = request.files['image']
    reference_width = float(request.form['width'].strip())

    # Validate input
    if file.filename == '':
        return jsonify({"error": "Image not uploaded"}), 400
    elif not isinstance(reference_width, float):
        return jsonify({"error": "Reference width not provided or invalid format"}), 400

    # Convert the uploaded image file to an OpenCV image
    orig_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Scale down image for faster processing
    # scaled_image = orig_image
    scaled_image = scale_down_image(orig_image)
    
    noBG = removeBG(scaled_image)
    if noBG is None:
        return jsonify({"error", "Failed to remove background"}), 400
        
    # Detect coin in BG-removed image, if none present, default to orig img
    cv2.imwrite("processed/removed_contrasted_image.jpg", noBG)



    coin_detected = detect_coin_contour(noBG)
    print("detected a coin:", coin_detected)
    finger_mask, hand_label = trackFinger(noBG)
    
    cv2.imwrite('processed/finger_mask.png', finger_mask)
    
    calSizeImg = None
    
    if coin_detected is False:
        calSizeImg = scaled_image
    else:
        calSizeImg = noBG
        
    cv2.imwrite('processed/temp.jpg', calSizeImg)
        
    data, processed_image = calsize.sizeCalculateFingers(calSizeImg, finger_mask, hand_label, reference_width)
    print('HAND LABEL {}'.format(hand_label))
    
    cv2.imwrite('processed/processed_image_fingers.jpg', processed_image)
    
    # print(response)
    
    img_io = BytesIO()
    processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    processed_pil_image.save(img_io, format='JPEG')
    img_io.seek(0)

    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    response = {
        "hand_label": hand_label,
        "finger_measurement": data,
        "processed_image": img_base64
    }

    return jsonify(response), 200

@app.route('/measure-wrist', methods=['POST'])
def measure_wrist():
    if 'image' not in request.files or 'width' not in request.form:
        return jsonify({"error": "No image file or width provided"}), 400

    file = request.files['image']
    reference_width = float(request.form['width'].strip())

    # Validate input
    if file.filename == '':
        return jsonify({"error": "Image not uploaded"}), 400
    elif not isinstance(reference_width, float):
        return jsonify({"error": "Reference width not provided or invalid format"}), 400

    # Convert the uploaded image file to an OpenCV image
    orig_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Scale down image for faster processing
    scaled_image = scale_down_image(orig_image)
    
    noBG = removeBG(scaled_image)
    if noBG is None:
        return jsonify({"error", "Failed to remove background"}), 400
        
    # Detect coin in BG-removed image, if none present, default to orig img

    coin_detected = detect_coin_contour(noBG)
    print("detected a coin:", coin_detected)
    wrist_mask, hand_label = trackWrist(noBG)
    
    #cv2.imwrite('processed/wrist_mask.png', wrist_mask)
    
    calSizeImg = None
    
    if coin_detected is False:
        calSizeImg = scaled_image
    else:
        calSizeImg = noBG
    
    data, processed_image = calsize.sizeCalculateWrist(calSizeImg, wrist_mask, hand_label, reference_width)
    print('HAND LABEL {}'.format(hand_label))
    
    #cv2.imwrite('processed/processed_image_wrist.jpg', processed_image)
    
    # print(response)
    
    img_io = BytesIO()
    processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    processed_pil_image.save(img_io, format='JPEG')
    img_io.seek(0)

    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    response = {
        "hand_label": hand_label,
        "wrist_measurement": data,
        "processed_image": img_base64
    }

    return jsonify(response), 200

if __name__ == "__main__":
    #app.run()
    app.run(host='0.0.0.0', port=8080, debug=True)