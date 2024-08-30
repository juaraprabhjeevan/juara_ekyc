# Juara eKYC Library

Juara eKYC is a Python library for electronic Know Your Customer (eKYC) verification, including document verification, face processing, liveness detection, and face matching.

## Features

- Document verification
- Face processing
- Liveness check
- Face matching
- Flask-based API for eKYC verification

## Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- deepface
- paddleocr
- Flask
- dlib

## Installation

### For Windows Users:

1. Ensure you have CMake installed. You can download it from [cmake.org](https://cmake.org/download/).

2. Uninstall any previous versions:
   ```
   pip uninstall ekyc
   ```

3. Clear pip cache:
   ```
   pip cache purge
   ```

4. Install the package:
   ```
   pip install path/to/ekyc-0.0.4-py3-none-any.whl
   ```

   Note: This will automatically install the correct dlib version for your Python installation.

5. You may need to install PaddleOCR and PaddlePaddle separately:
   ```
   pip install paddlepaddle
   pip install paddleocr
   ```

### For Other Operating Systems:

You can install the eKYC library using pip:

pip install juara_ekyc


## Usage

Here's a basic example of how to use the Juara eKYC library:

python
from juara_ekyc import process_id_verification
result, message = process_id_verification('path/to/image.jpg')
print(f"Verification result: {result}")
print(f"Message: {message}")

## Development

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/juara_ekyc.git
   cd juara_ekyc
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the development dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the tests:
   ```
   python -m unittest discover tests
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.