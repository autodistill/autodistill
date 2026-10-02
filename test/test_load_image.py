from autodistill.helpers import load_image

from PIL import Image
import cv2
import numpy as np
import os
import tempfile

TEST_IMAGE = "test/data/dog.jpeg"

pil_image = Image.open(TEST_IMAGE)
cv2_image = cv2.imread(TEST_IMAGE)
np_image = np.array(pil_image)
string_image = TEST_IMAGE
url = "https://media.roboflow.com/dog.jpeg"

assert isinstance(load_image(pil_image, return_format="PIL"), Image.Image)
assert isinstance(load_image(pil_image, return_format="cv2"), np.ndarray)
assert isinstance(load_image(pil_image, return_format="numpy"), np.ndarray)

assert isinstance(load_image(cv2_image, return_format="PIL"), Image.Image)
assert isinstance(load_image(cv2_image, return_format="cv2"), np.ndarray)
assert isinstance(load_image(cv2_image, return_format="numpy"), np.ndarray)

assert isinstance(load_image(np_image, return_format="PIL"), Image.Image)
assert isinstance(load_image(np_image, return_format="cv2"), np.ndarray)
assert isinstance(load_image(np_image, return_format="numpy"), np.ndarray)

assert isinstance(load_image(string_image, return_format="PIL"), Image.Image)
assert isinstance(load_image(string_image, return_format="cv2"), np.ndarray)
assert isinstance(load_image(string_image, return_format="numpy"), np.ndarray)

assert isinstance(load_image(url, return_format="PIL"), Image.Image)
assert isinstance(load_image(url, return_format="cv2"), np.ndarray)
assert isinstance(load_image(url, return_format="numpy"), np.ndarray)

# Test PPM support if test_ppm_support is available
try:
    from test_ppm_support import create_test_ppm
    
    # Create a test PPM file
    TEST_PPM = create_test_ppm()
    
    # Test PPM loading
    assert isinstance(load_image(TEST_PPM, return_format="PIL"), Image.Image)
    assert isinstance(load_image(TEST_PPM, return_format="cv2"), np.ndarray)
    assert isinstance(load_image(TEST_PPM, return_format="numpy"), np.ndarray)
    
    # Clean up
    if os.path.exists(TEST_PPM):
        os.remove(TEST_PPM)
        
    print("PPM support tests passed!")
except (ImportError, ModuleNotFoundError):
    print("Skipping PPM tests - test_ppm_support module not found")

# Test error handling for non-image file
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
    tmp.write("This is not an image.")
    tmp_path = tmp.name
try:
    try:
        load_image(tmp_path)
        assert False, "Expected ValueError for non-image file"
    except ValueError as e:
        assert "not a valid image" in str(e)
finally:
    os.remove(tmp_path)