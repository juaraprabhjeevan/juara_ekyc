from setuptools import setup, find_packages
import sys
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Determine the appropriate dlib wheel based on Python version
python_version = f"{sys.version_info.major}{sys.version_info.minor}"
dlib_wheel = f"dlib-19.22.99-cp{python_version}-cp{python_version}-win_amd64.whl"
dlib_wheel_path = os.path.join("wheels", dlib_wheel)

# Check if the wheel exists
if not os.path.exists(dlib_wheel_path):
    raise RuntimeError(f"Dlib wheel not found for Python {python_version}. Please check the 'wheels' directory.")

setup(
    name="ekyc",
    version="0.0.1",
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
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "deepface>=0.0.75",
        "paddleocr>=2.0.6",
        "paddlepaddle>=2.0.0",  # Add this line to ensure PaddlePaddle is installed
        "Flask>=2.0.0",
        f"dlib @ file:///{os.path.abspath(dlib_wheel_path)}",
    ],
    package_data={
        'ekyc': [
            'data/ic_template.jpg',
            'data/shape_predictor_68_face_landmarks.dat',
        ],
    },
    include_package_data=True,
    data_files=[('wheels', [dlib_wheel_path])],
)