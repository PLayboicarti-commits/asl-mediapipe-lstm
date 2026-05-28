import cv2
import mediapipe as mp
import numpy as np
from utils import create_directory, extract_frame_keypoints

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 30


def collect_data(sign_name, num_sequences=50):
    create_directory("data")
    create_directory(f"data/{sign_name}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cv2.namedWindow("ASL Data Collection", cv2.WINDOW_NORMAL)

    # Warm up webcam
    for _ in range(5):
        cap.read()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        print(f"Collecting sequences for: {sign_name}")
        print("Press SPACE to start each sequence")
        print("Press Q to quit")

        sequence_count = 0

        while sequence_count < num_sequences:
            # Waiting screen
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)

                cv2.putText(frame, f"Sign: {sign_name}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Sequence {sequence_count + 1}/{num_sequences}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to record 30 frames", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow("ASL Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            sequence = []

            for frame_idx in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                keypoints = extract_frame_keypoints(results)
                sequence.append(keypoints)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                cv2.putText(frame, f"Recording {sign_name}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Sequence {sequence_count + 1}/{num_sequences}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame {frame_idx + 1}/{SEQUENCE_LENGTH}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow("ASL Data Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            sequence = np.array(sequence, dtype=np.float32)
            np.save(f"data/{sign_name}/{sign_name}_seq_{sequence_count}.npy", sequence)
            print(f"Saved sequence {sequence_count + 1}/{num_sequences} for {sign_name}")

            sequence_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sign_name = input("Enter ASL sign name (e.g., HELLO, THANK_YOU): ").strip().upper()
    num_sequences = int(input("Enter number of sequences to collect: "))
    collect_data(sign_name, num_sequences)