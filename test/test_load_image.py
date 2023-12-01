from autodistill.helpers import load_image

from PIL import Image
import cv2
import numpy as np

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