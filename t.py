import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # Add face mesh for drawing face landmarks

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

# Streamlit UI
st.title("Sign Language Recognition")

# Label input
label = st.text_input("Enter label for the sign")
if label:
    # Create a directory for the label in the current working directory
    label_path = os.path.join(os.getcwd(), label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    # Buttons for start and stop recording
    start_button = st.button("Start Recording")
    stop_button = st.button("Stop Recording")
    
    # Initialize recording variables
    recording = False
    sequence = []
    sequences = []
    labels = []

    # Video capture setup
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])  # Create an empty placeholder for the webcam feed

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:  # Use face mesh here
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)  # Use face mesh connections here
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Start recording when the button is pressed
            if start_button:
                recording = True
                st.write("Recording started...")

            # Stop recording when the button is pressed
            if stop_button:
                recording = False
                sequences.append(sequence)
                labels.append(label)  # Add label for the current sequence
                sequence = []
                st.write("Recording stopped.")
                break  # Stop the loop after stopping the recording

            if recording:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

            # Display the webcam feed in Streamlit
            frame_window.image(image, channels="BGR", use_column_width=True)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()

    # Save Data Automatically After Collection
    for sequence_idx, seq in enumerate(sequences):
        npy_path = os.path.join(label_path, f"sequence_{sequence_idx}.npy")
        np.save(npy_path, seq)
    
    # Save the label map
    with open(os.path.join(os.getcwd(), "labels.pkl"), "wb") as f:
        pickle.dump({label: len(labels)}, f)

    st.success("Data saved successfully!")

# Option to train the model
train_button = st.button("Train Model")
if train_button:
    # Ensure that labels are not empty before proceeding
    if not labels:
        st.error("No data available for training. Please collect data first.")
    else:
        data, labels = [], []
        for label in os.listdir(os.getcwd()):
            if os.path.isdir(os.path.join(os.getcwd(), label)):
                for sequence in os.listdir(os.path.join(os.getcwd(), label)):
                    if sequence.endswith(".npy"):
                        npy_path = os.path.join(os.getcwd(), label, sequence)
                        data.append(np.load(npy_path))
                        labels.append(label)

        # Debugging: Check if labels are populated correctly
        st.write(f"Labels: {labels}")

        if labels:
            X = np.array(data)
            y = to_categorical(labels).astype(int)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = Sequential([
                LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
                LSTM(128, return_sequences=False, activation='relu'),
                Dense(64, activation='relu'),
                Dense(len(set(labels)), activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
            model.save(os.path.join(os.getcwd(), "model.h5"))
            st.success("Model trained and saved successfully!")
