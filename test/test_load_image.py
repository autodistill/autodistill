from autodistill.helpers import load_image

from PIL import Image
import cv2
import numpy as np

pil_image = Image.open("dog.jpeg")
cv2_image = cv2.imread("dog.jpeg")
np_image = np.array(pil_image)
string_image = "data/dog.jpeg"
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