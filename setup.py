import re

import setuptools
from setuptools import find_packages

with open("./autodistill/__init__.py", "r") as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodistill",
    version=version,
    author="Roboflow",
    author_email="autodistill@roboflow.com",
    description="Distill large foundational models into smaller, domain-specific models for deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autodistill/autodistill",
    install_requires=[
        "opencv-python>=4.6.0",
        "supervision",
        "tqdm",
        "Pillow>=7.1.2",
        "PyYAML>=5.3.1",
        "click"
    ],
    # allow models.csv in package
    include_package_data=True,
    package_data={"": ["models.csv"]},
    packages=find_packages(exclude=("tests",)),
    entry_points={
        "console_scripts": [
            "autodistill=autodistill.cli:main",
        ],
    },
    extras_require={
        "dev": [
            "flake8",
            "black==22.3.0",
            "isort",
            "twine",
            "pytest",
            "wheel",
            "mkdocs-material",
            "mkdocs",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
