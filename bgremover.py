from rembg import remove 
import numpy as np
from PIL import Image

class BackgroundRemover:
    def __init__(self, input_image):
        if isinstance(input_image, Image.Image):
            self.input_image = input_image
        else:
            raise ValueError("input_image must be a valid image object.")

    def process_image(self):
        try:
            
            # Remove the background
            imageWithoutBg = remove(self.input_image)
            
            # Return the processed image
            return imageWithoutBg
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
