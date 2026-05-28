import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from tensorflow.keras.models import load_model
from utils import extract_frame_keypoints

SEQUENCE_LENGTH = 30
MODEL_PATH = "models/asl_sequence_model.h5"
CLASS_NAMES_PATH = "models/class_names.npy"
CONFIDENCE_THRESHOLD = 0.80

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open camera first so the window appears sooner
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cv2.namedWindow("ASL Recognition", cv2.WINDOW_NORMAL)

# Warm up webcam
for _ in range(5):
    cap.read()

model = load_model(MODEL_PATH)
class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=10)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        keypoints = extract_frame_keypoints(results)
        sequence_buffer.append(keypoints)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        predicted_sign = "No hands detected"
        confidence = 0.0

        if results.multi_hand_landmarks:
            predicted_sign = "Collecting frames..."

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence_buffer, dtype=np.float32), axis=0)
                prediction = model.predict(input_data, verbose=0)[0]

                predicted_class = np.argmax(prediction)
                confidence = float(prediction[predicted_class])

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(class_names[predicted_class])
                    predicted_sign = Counter(prediction_history).most_common(1)[0][0]
                else:
                    predicted_sign = "Uncertain"
        else:
            prediction_history.clear()

        cv2.putText(frame, f"Prediction: {predicted_sign}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Frames: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("ASL Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()