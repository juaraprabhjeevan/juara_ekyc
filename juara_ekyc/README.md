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

You can install the Juara eKYC library using pip:

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