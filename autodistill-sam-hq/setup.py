import setuptools
from setuptools import find_packages
import re

with open("./autodistill_sam_hq/__init__.py", 'r') as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)
    
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodistill-sam-hq",
    version=version,
    author="Roboflow",
    author_email="support@roboflow.com",
    description="Segment Anything High Quality (SAM HQ) model for use with Autodistill",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autodistill/autodistill-sam-hq",
    install_requires=[
        "torch",
        "autodistill",
        "numpy>=1.20.0",
        "opencv-python>=4.6.0",
        "rf_segment_anything",
        "supervision"
    ],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
