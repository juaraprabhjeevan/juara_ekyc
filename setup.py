from setuptools import setup, find_packages
import sys
import os

# Read the version from setup.cfg
from configparser import ConfigParser
config = ConfigParser()
config.read('setup.cfg')
version = config['metadata']['version']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Determine if we're on Windows
is_windows = sys.platform.startswith('win')

# Base requirements
install_requires = [
    "opencv-python>=4.5.0",
    "numpy>=1.19.0",
    "scikit-learn>=0.24.0",
    "deepface>=0.0.93",  # Updated to require the latest version 0.0.93
    "Flask>=2.0.0",
]

# Add paddleocr and paddlepaddle for non-Windows platforms
if not is_windows:
    install_requires.extend([
        "paddleocr>=2.0.6",
        "paddlepaddle>=2.0.0",
    ])

# Handle dlib installation
if is_windows:
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    dlib_wheel = f"dlib-19.22.99-cp{python_version}-cp{python_version}-win_amd64.whl"
    dlib_wheel_path = os.path.join("wheels", dlib_wheel)
    if os.path.exists(dlib_wheel_path):
        install_requires.append(f"dlib @ file:///{os.path.abspath(dlib_wheel_path)}")
    else:
        print(f"Warning: Dlib wheel not found for Python {python_version}. You may need to install dlib manually.")
else:
    install_requires.append("dlib>=19.22.0")

setup(
    name="ekyc",
    version="0.0.5",  # Update this line
    author="Prabhjeevan Singh",
    author_email="prabhjeevan@juarapartners.com",
    description="A library for electronic Know Your Customer (eKYC) verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juaraprabhjeevan/juara_ekyc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    package_data={
        'ekyc': [
            'data/ic_template.jpg',
            'data/shape_predictor_68_face_landmarks.dat',
        ],
    },
    include_package_data=True,
)