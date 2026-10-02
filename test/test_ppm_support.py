import os
import sys
import numpy as np
import cv2
from PIL import Image

# Add parent directory to path so we can import autodistill
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodistill.helpers import load_image

def create_test_ppm(filepath="test/data/test.ppm"):
    """Create a simple PPM image file for testing"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create a simple 100x100 colored image
    width, height = 100, 100
    
    # PPM P6 format (binary)
    with open(filepath, 'wb') as f:
        f.write(b'P6\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'255\n')  # Max color value
        
        # Write RGB data
        for y in range(height):
            for x in range(width):
                # Create a gradient
                r = (x * 255) // width
                g = (y * 255) // height
                b = ((x+y) * 255) // (width+height)
                f.write(bytes([r, g, b]))
    
    return filepath

def test_ppm_loading():
    """Test that PPM files can be loaded correctly using the load_image function"""
    # Create a test PPM file
    test_file = create_test_ppm()
    
    # Test loading with various return formats
    pil_image = load_image(test_file, return_format="PIL")
    assert isinstance(pil_image, Image.Image), "Failed to load PPM as PIL Image"
    
    cv2_image = load_image(test_file, return_format="cv2")
    assert isinstance(cv2_image, np.ndarray), "Failed to load PPM as cv2 image"
    assert len(cv2_image.shape) == 3, "PPM should be loaded as a 3-channel image"
    
    numpy_image = load_image(test_file, return_format="numpy")
    assert isinstance(numpy_image, np.ndarray), "Failed to load PPM as numpy array"
    assert len(numpy_image.shape) == 3, "PPM should be loaded as a 3-channel image"
    
    print("All PPM loading tests passed!")
    
    # Clean up
    os.remove(test_file)

if __name__ == "__main__":
    test_ppm_loading()
