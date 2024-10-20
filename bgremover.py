# bgremover.py
from rembg import remove 
import numpy as np
from PIL import Image
import os

class BackgroundRemover:
    def __init__(self, input_path):
        self.input_path = input_path

    def process_image(self):
        
        # Get the file extension
        file_extension = os.path.splitext(self.input_path)[1][1:].lower()

        # Validate file type
        if file_extension not in ['jpg', 'jpeg', 'png']:
            print(f"Warning: Unsupported file type '{file_extension}'. Supported file types are: jpg, jpeg, png.")
            exit()
        
        try:
            # Load the image
            orig = Image.open(self.input_path)
            
            # Convert the input image to a numpy array
            input_image_np = np.array(orig)
            
            # Remove the background
            imageWithoutBg = remove(input_image_np)
            
            # Create a PIL Image from the output array
            output_image = Image.fromarray(imageWithoutBg)
            
            # Return the processed image
            return output_image
        
        except Exception as e:
            return str(f"An error occurred: {e}")