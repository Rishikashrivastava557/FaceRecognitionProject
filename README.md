# Face Recognition Attendance System

A Python-based face recognition application that uses computer vision and PCA-based Eigenfaces for real-time face detection and recognition.

## Features

* Real-time face detection using OpenCV
* Face image capture through webcam
* PCA-based Eigenfaces for face recognition
* Real-time face recognition
* Tkinter-based graphical user interface
* Organized training image management

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Tkinter
* Pillow

## Project Structure

```text
FaceRecognitionProject/
│
├── face_recognition_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── trainingImages/
└── eigenfaces/
```

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python face_recognition_app.py
```

## How It Works

1. Capture face images using the webcam.
2. Store training images in the training directory.
3. Build Eigenfaces using PCA.
4. Start real-time face recognition.
5. The system compares detected faces with the training images.

## Future Improvements

* Automatic attendance recording with date and time
* CSV-based attendance management
* Multiple-person identification
* Improved recognition accuracy
* Enhanced Tkinter dashboard
