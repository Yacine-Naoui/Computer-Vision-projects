# Computer Vision Projects

This repository contains two computer vision projects:

## 1. HandWritten Digit Recognition

**Overview:**  
This project is a simple application that allows you to draw digits on the screen using your mouse. The drawing is captured by Pygame and processed by a pre-trained neural network built with Keras. The model, trained on the MNIST dataset, classifies the drawn digit and displays the recognized number on the screen.

**Key Features:**
- **Drawing Interface:**  
  Use your mouse to draw digits directly on the application window.
- **Real-Time Prediction:**  
  When you finish drawing, the app processes the image, resizes and pads it as needed, and then uses the Keras model to predict the digit.
- **Optional Image Saving:**  
  The app supports saving the drawn image for further review or analysis.
- **Clear Screen Functionality:**  
  Easily clear the canvas by pressing the `N` key.

**Technologies Used:**
- **Pygame:** For creating the graphical user interface and capturing user input.
- **Keras/TensorFlow:** For building and running the neural network model.
- **OpenCV & NumPy:** For image processing tasks such as resizing and padding.

## 2. Object Detection App

**Overview:**  
This project is an interactive object detection application that leverages YOLOv5 for detecting objects within images. The app uses Kivy to provide a user-friendly graphical interface that lets you load an image, adjust the detection confidence, and view the results with bounding boxes and labels.

**Key Features:**
- **Image Loading:**  
  Load images from your file system using a built-in file chooser.
- **YOLOv5 Integration:**  
  The app uses a pre-trained YOLOv5 model to detect objects in the image.
- **Adjustable Confidence Threshold:**  
  Use a slider to set the minimum confidence level for displayed detections.
- **Result Visualization:**  
  Displays the processed image with bounding boxes and labels, and also provides a detailed results table listing detected objects, their confidence scores, and bounding box coordinates.

**Technologies Used:**
- **Kivy:** For building the multi-touch user interface.
- **PyTorch:** For running the YOLOv5 model.
- **OpenCV & NumPy:** For image processing and drawing bounding boxes.
- **YOLOv5:** A state-of-the-art object detection model from the Ultralytics repository.

---

## Setup and Installation

### Prerequisites

- **Python 3.11** (It is recommended to use a Python 3.11 virtual environment.)
- pip

### Creating a Python 3.11 Virtual Environment

1. Navigate to the repository directory:
   ```bash
   cd /path/to/repository
   ```
2. Create a virtual environment:
    ```bash
    python3.11 -m venv venv
    ```

3. Activate the virtual environment:
    - **On Linux/macOS:**
      ```bash
      source venv/bin/activate
      ```
    - **On Windows:**
      ```bash
      venv\Scripts\activate
      ```

4. Install all required dependencies using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

*This will install all the necessary packages for both the HandWritten Digit Recognition and Object Detection App projects.*

## Requirements

The `requirements.txt` file includes the following dependencies:

```txt
# Common dependencies
numpy
opencv-python
matplotlib

# For HandWritten Digit Recognition Project
pygame
keras
tensorflow  # Backend for Keras

# For Object Detection App
torch
torchvision
torchaudio
ultralytics
kivy
