# Facial Emotion Recognition

## Project Description

This project is an implementation of facial emotion recognition using Python, leveraging OpenCV for face detection and TensorFlow for recognizing facial expressions. The program displays video output from the webcam with bounding boxes around detected faces and labels indicating the recognized facial expressions.

## Prerequisites

Before running the program, make sure you have installed the following libraries:

- OpenCV: `pip install opencv-python`
- TensorFlow: `pip install tensorflow`
- Flask: `pip install flask`

## Usage

1. Open a terminal or command prompt.
2. Navigate to the directory where the Python file (`facial_emotion_recognition.py`) is located.
3. Run the program with the command:

   ```bash
   python facial_emotion_recognition.py

Open your web browser and navigate to http://127.0.0.1:5000/. You should see the webcam feed with Real-Time face detection and expression recognition.



## Directory Structure
- haarcascades: Directory containing cascade files for face detection.
- path/to/your/expression_model.h5: Location where you store the facial expression model (replace with the appropriate path).
- face_recognition.py: Main source code of the program.

## Notes
- Ensure you have Python and all the required libraries installed.
- Make sure the webcam is available and connected to the computer.
- The facial expression model should be adjusted and saved in the correct path before running the program.