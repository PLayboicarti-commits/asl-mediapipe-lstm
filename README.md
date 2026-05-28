Real-Time ASL Gesture Recognition

A deep learning-based system for recognizing American Sign Language (ASL) gestures in real time using hand landmark sequences and temporal modeling.

About the Project

This project was developed as a final-year major project with the goal of improving accessibility in communication.

The system uses a webcam to capture live hand movements and converts them into text in real time. Unlike traditional approaches that rely on image or video datasets, this project uses hand landmark sequences extracted using MediaPipe.

By modeling gestures as time-based sequences, the system captures both spatial and temporal information, leading to more accurate recognition of dynamic signs.

Features
Real-time ASL recognition via webcam
Trained on landmark sequences (.npy), not images or videos
Uses LSTM-based temporal modeling to capture motion
Supports detection of up to 2 hands
Landmark normalization for robustness to distance and position
Smooth prediction using sequence buffering
Easy to retrain or extend with new gestures
Tech Stack
Python
OpenCV
MediaPipe
TensorFlow / Keras
NumPy
Scikit-learn
Tkinter (optional GUI)
Folder Structure
Real-Time-ASL-Gesture-Recognition/
├── data/                # Sequence training data (.npy)
│   ├── HELLO/
│   ├── YES/
│   └── ...
├── models/
│   ├── asl_sequence_model.h5   # Trained LSTM model
│   ├── class_names.npy         # Gesture labels
│   └── metadata.json           # Input configuration
├── src/
│   ├── data_collection.py      # Collect training sequences
│   ├── train_model.py          # Train LSTM model
│   ├── asl_recognition.py      # Real-time inference
│   ├── utils.py                # Helper functions
│   └── asl_gui.py              # (Optional GUI)
├── requirements.txt
├── README.md
└── LICENSE
Model Input Format
Sequence length: 30 frames
Features per frame: 126
(2 hands × 21 landmarks × 3 coordinates)
Input shape:
(30, 126)

Each gesture is represented as a sequence of normalized landmark positions over time, allowing the model to learn motion patterns instead of static poses.

How It Works
The webcam captures live video input
MediaPipe detects up to 2 hands and extracts 21 landmarks per hand
Landmark coordinates are normalized relative to the wrist
A rolling buffer collects 30 consecutive frames
The sequence is passed into an LSTM neural network
The model predicts the ASL gesture
The prediction is displayed in real time
Getting Started
1. Clone the repository
git clone https://github.com/kaushiks-info/Real-Time-ASL-Gesture-Recognition.git
cd Real-Time-ASL-Gesture-Recognition
2. Install dependencies
pip install -r requirements.txt
3. Collect your own training data
python src/data_collection.py
Enter a gesture name (e.g., HELLO, YES)
Record multiple sequences per gesture
Each sequence = 30 frames
4. Train the model
python src/train_model.py

This will create:

models/asl_sequence_model.h5
models/class_names.npy
5. Run real-time recognition
python src/asl_recognition.py
Important Note – Model File

The trained model (asl_sequence_model.h5) is not included in this repository.

This is intentional because recognition accuracy depends heavily on:

your hand shape
lighting conditions
camera setup

Training your own model ensures better performance and personalization.

Limitations
Performance may drop if hands are partially blocked (occlusion)
Requires both hands to be visible for some gestures
Sensitive to lighting and background conditions
Works best when gestures are clearly centered in the frame
Future Work
Improve robustness to occlusion and fast motion
Add sentence generation from continuous gestures
Expand dataset with multiple users
Improve model architecture (e.g., Transformer-based models)
Integrate a full GUI application
Preview
<h3>ASL Gesture Recognition - Screenshot</h3> <img src="https://github.com/user-attachments/assets/5582e240-6585-4032-9ffd-f960d2aef9af" width="500"> <img src="https://github.com/user-attachments/assets/726b8d8d-fa00-459f-b992-eb113515ecc2" width="300">
License

This project is licensed under the MIT License.

Summary

This project demonstrates a real-time sign language recognition system using sequence-based landmark data and LSTM modeling, trained directly from webcam input without relying on external video datasets.