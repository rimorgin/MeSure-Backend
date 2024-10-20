import cv2
import argparse
from bgremover import BackgroundRemover
import numpy as np
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

orig_image = args['image']

# Create an instance of BackgroundRemover
remover = BackgroundRemover(orig_image)

try:
    # Process the image and get the result as a PIL Image
    output_image = remover.process_image()

    # Convert PIL Image back to NumPy array for OpenCV if needed
    output_image_np = np.array(output_image)
    
    # Extract filename without extension and save as .png
    #output_filename = os.path.basename(orig_image)
    
    filename = os.path.splitext(orig_image)[0]
    file_ext = os.path.splitext(orig_image)[1]
    output_filename = f'{filename}BG{file_ext}'
    
    # Check if the image has an alpha channel (RGBA)
    if output_image.mode == 'RGBA':
        output_image = output_image.convert('RGB')  # Convert to RGB if necessary
    
    # Save the output image using OpenCV
    output_image.save(output_filename)

    # Load the image in OpenCV
    removedBG = cv2.imread(output_filename)

    # Check if the image was loaded correctly
    if removedBG is not None:
        # Display the output image
        cv2.imshow('removedBG', removedBG)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
        os.remove(output_filename)
    else:
        print("Error: Could not load the image for display.")

except Exception as e:
    print(e)
