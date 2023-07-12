import setuptools
from setuptools import find_packages
import re

with open("./autodistill/__init__.py", 'r') as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)
    
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().split('\n')

setuptools.setup(
    name="autodistill",  
    version=version,
    author="Roboflow",
    author_email="autodistill@roboflow.com",
    description="Distill large foundational models into smaller, domain-specific models for deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autodistill/autodistill",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel", "mkdocs-material", "mkdocs"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License :: 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
