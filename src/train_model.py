import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data"
MODEL_DIR = "models"
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 126


def load_sequence_data(data_dir):
    X = []
    y = []

    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)

        if not os.path.isdir(label_dir):
            continue

        for file_name in os.listdir(label_dir):
            if not file_name.endswith(".npy"):
                continue

            file_path = os.path.join(label_dir, file_name)
            sequence = np.load(file_path)

            if sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                print(
                    f"Skipping {file_path}, expected "
                    f"{(SEQUENCE_LENGTH, FEATURES_PER_FRAME)} but got {sequence.shape}"
                )
                continue

            X.append(sequence)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y


def main():
    X, y = load_sequence_data(DATA_DIR)

    if len(X) == 0:
        raise ValueError("No valid sequence data found in 'data/'")

    print("Loaded dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    class_names = label_encoder.classes_

    if len(class_names) < 2:
        raise ValueError(
            "You need at least 2 gesture classes to train a useful model. "
            "Collect more classes like HELLO, YES, NO, etc."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test, y_test)
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # ===== Evaluation Metrics =====
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ===== Save Model =====
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "asl_sequence_model.h5"))
    np.save(os.path.join(MODEL_DIR, "class_names.npy"), class_names)

    # ===== Save Training Graphs =====
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(MODEL_DIR, 'training_accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(MODEL_DIR, 'training_loss.png'))
    plt.close()

    print("\nSaved files:")
    print(os.path.join(MODEL_DIR, "asl_sequence_model.h5"))
    print(os.path.join(MODEL_DIR, "class_names.npy"))
    print(os.path.join(MODEL_DIR, "training_accuracy.png"))
    print(os.path.join(MODEL_DIR, "training_loss.png"))

    print("\nSequence model trained and saved!")


if __name__ == "__main__":
    main()