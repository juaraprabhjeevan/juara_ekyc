from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ekyc_lib",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for electronic Know Your Customer (eKYC) verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ekyc_lib",
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
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "deepface>=0.0.75",
        "paddleocr>=2.0.6",
        "Flask>=2.0.0",
        "dlib>=19.22.0",
    ],
    package_data={
        'ekyc_lib': ['shape_predictor_68_face_landmarks.dat'],
    },
    include_package_data=True,
)