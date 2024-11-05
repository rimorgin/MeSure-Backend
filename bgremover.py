from rembg import remove 
import numpy as np
from PIL import Image

class BackgroundRemover:
    def __init__(self, input_image):
        if isinstance(input_image, Image.Image):
            self.input_image = input_image
        else:
            raise ValueError("input_image must be a PIL Image object.")

    def process_image(self):
        try:
            # Convert the input image to a numpy array
            input_image_np = np.array(self.input_image)
            
            # Remove the background
            imageWithoutBg = remove(input_image_np)
            
            # Create a PIL Image from the output array
            output_image = Image.fromarray(imageWithoutBg)
            
            # Return the processed image
            return output_image
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

# Example usage:
# image = Image.open('path/to/image.jpg')  # Load your image as a PIL Image
# bg_remover = BackgroundRemover(input_image=image)
# processed_image = bg_remover.process_image()