import cv2
import numpy as np

class ImageOverlay:
    def __init__(self, background_image: np.ndarray, overlay_image: np.ndarray, overlay_size_mm: float, ppi: int):
        # Expecting images as NumPy arrays directly
        self.background = background_image
        self.overlay = overlay_image
        self.overlay_size_mm = overlay_size_mm
        self.ppi = ppi

        if self.background is None:
            raise ValueError("Background image cannot be None")
        if self.overlay is None:
            raise ValueError("Overlay image cannot be None")

        # Calculate new size in pixels
        self.overlay_width, self.overlay_height = self.calculate_overlay_size(overlay_size_mm, ppi)

        # Resize the overlay image
        self.overlay = cv2.resize(self.overlay, (self.overlay_width, self.overlay_height), interpolation=cv2.INTER_AREA)

    def calculate_overlay_size(self, overlay_size_mm: float, ppi: int):
        # Convert millimeters to inches and then to pixels
        size_inch = overlay_size_mm / 25.4  # 1 inch = 25.4 mm
        size_pixels = size_inch * ppi
        return int(size_pixels), int(size_pixels)  # Return as (width, height)

    def add_overlay(self, position: tuple):
        # Add overlay to background at specified position
        overlay_height, overlay_width = self.overlay.shape[:2]
        x, y = position
        roi = self.background[y:y + overlay_height, x:x + overlay_width]

        if self.overlay.shape[2] == 4:  # Check for alpha channel
            b, g, r, a = cv2.split(self.overlay)
            overlay_mask = a / 255.0
            overlay_mask_inv = 1 - overlay_mask
            for c in range(3):  # Blend colors
                roi[:, :, c] = (overlay_mask * self.overlay[:, :, c] + overlay_mask_inv * roi[:, :, c])
        else:
            roi[:] = self.overlay  # No alpha, just overlay

        self.background[y:y + overlay_height, x:x + overlay_width] = roi
        return self.background

"""
# Usage example:
if _name_ == "_main_":
    background_image_path = 'path/to/background.png'  # Path to the background image
    overlay_image_path = 'path/to/overlay.png'        # Path to the overlay image
    output_image_path = 'path/to/output.png'          # Path to save the result

    # Create an instance of the ImageOverlay class with size in mm and PPI
    overlay_size_mm = 23  # Size of overlay in millimeters
    ppi = 96              # Change to 300 for high-resolution size

    image_overlay = ImageOverlay(background_image_path, overlay_image_path, overlay_size_mm, ppi)

    # Specify the position to place the overlay (e.g., top left corner)
    position = (0, 0)

    # Add the overlay and save the result
    image_overlay.add_overlay(position, output_image_path)

    print(f"Overlay added and saved to {output_image_path}")
"""