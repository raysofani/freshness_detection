# Freshness Detection Application

## Project Overview
This is a machine learning-powered web application that detects the freshness of fruits and vegetables using computer vision and deep learning techniques.

## Features
- File upload for image analysis
- Live camera detection
- Machine learning-based freshness prediction
- User-friendly web interface
- Supports multiple image formats

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/raysofani/freshness_detection.git
cd freshness_detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model and Label Preparation
- Ensure `keras_Model.h5` is in the project root
- Verify `labels.txt` exists with your class labels
  - Format: One label per line
  - Example:
    ```
    Fresh Apple
    Slightly Ripe
    Overripe
    ```

## Running the Application

### Start the Flask Server
```bash
# On Windows
python app.py

# On macOS/Linux
python3 app.py
```

### Access the Application
- Open a web browser
- Navigate to `http://localhost:5000`

## Usage Modes

### 1. File Upload
- Click "File Upload" tab
- Select an image
- Click "Predict Freshness"

### 2. Live Camera
- Click "Live Camera" tab
- Click "Start Camera"
- Click "Capture Photo"

## Troubleshooting
- Verify all dependencies are installed
- Check model and label files
- Ensure webcam permissions are granted
- Maximum file upload: 16MB
- Supported formats: PNG, JPG, JPEG

## Model Training Tips
- Collect diverse produce images
- Create balanced dataset
- Use data augmentation
- Consider transfer learning

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## Technologies
- Python
- Flask
- TensorFlow
- Keras
- HTML5
- JavaScript

