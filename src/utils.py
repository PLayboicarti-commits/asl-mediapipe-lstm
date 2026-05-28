import os
import numpy as np

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS = 3
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS
FEATURES_PER_FRAME = NUM_HANDS * FEATURES_PER_HAND  # 126


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_hand_landmarks(hand_landmarks):
    """
    Normalize landmarks relative to wrist (landmark 0).
    Returns flattened shape (63,)
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )
    wrist = coords[0].copy()
    coords = coords - wrist
    return coords.flatten()


def extract_frame_keypoints(results):
    """
    Extract up to 2 hands from MediaPipe results.
    Output shape: (126,)
    If fewer than 2 hands are detected, pad with zeros.
    """
    frame_keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            frame_keypoints.append(normalize_hand_landmarks(hand_landmarks))

    while len(frame_keypoints) < 2:
        frame_keypoints.append(np.zeros(FEATURES_PER_HAND, dtype=np.float32))

    return np.concatenate(frame_keypoints).astype(np.float32)