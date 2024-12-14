import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving and loading the model

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Directory for storing data
data_dir = "C:\\Users\\manav\\OneDrive\\Desktop\\major\\data"

# Streamlit UI setup
st.set_page_config(page_title="Sign Language App", layout="wide")
st.title("Sign Language Recording and Training App")

# Menu bar with options
menu_option = st.sidebar.selectbox("Select Option", ["Record Data", "Train Model", "Use Model"])

# Record Data
if menu_option == "Record Data":
    label = st.text_input("Enter Action Label", "Enter the label here (e.g., hello, thanks)")
    start_button = st.button("Start Recording")
    stop_button = st.button("Stop Recording")

    if label:
        # Sanitize the label and ensure ASCII-compatible folder names
        folder_path = os.path.join(data_dir, label)
        folder_path = folder_path.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Placeholder for sequence count
        sequences_stored_placeholder = st.empty()
        sequences_stored = len([file for file in os.listdir(folder_path) if file.endswith(".npy")])
        sequences_stored_placeholder.write(f"Sequences stored for '{label}': **{sequences_stored}**")

    # Flag to control recording
    is_recording = False

    # Function to record and store data
    def record_sign_language(label):
        global is_recording

        cap = cv2.VideoCapture(0)
        frame_count = 0
        sequences_stored = len([file for file in os.listdir(folder_path) if file.endswith(".npy")])

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while is_recording:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam. Please check your camera.")
                    break

                # Flip and process the frame
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)

                # Draw landmarks on frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Save landmarks every 30 frames
                if frame_count % 30 == 0:
                    landmarks_data = []
                    if results.pose_landmarks:
                        for lm in results.pose_landmarks.landmark:
                            landmarks_data.append([lm.x, lm.y, lm.z])
                    if results.face_landmarks:
                        for lm in results.face_landmarks.landmark:
                            landmarks_data.append([lm.x, lm.y, lm.z])
                    if results.left_hand_landmarks:
                        for lm in results.left_hand_landmarks.landmark:
                            landmarks_data.append([lm.x, lm.y, lm.z])
                    if results.right_hand_landmarks:
                        for lm in results.right_hand_landmarks.landmark:
                            landmarks_data.append([lm.x, lm.y, lm.z])

                    # Save the numpy data and handle encoding issues
                    try:
                        np.save(os.path.join(folder_path, f"{sequences_stored}.npy"), np.array(landmarks_data), allow_pickle=True)
                    except Exception as e:
                        st.error(f"Error saving data: {e}")
                        break

                    sequences_stored += 1
                    sequences_stored_placeholder.write(f"Sequences stored for '{label}': **{sequences_stored}**")

                frame_count += 1
                frame_placeholder.image(frame, channels="BGR")

                if not is_recording:
                    break

        cap.release()

    # Placeholder for video and messages
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    # Start and stop recording
    if start_button and label:
        is_recording = True
        status_placeholder.info("Recording started. Press 'Stop Recording' to end.")
        record_sign_language(label)

    if stop_button:
        is_recording = False
        status_placeholder.warning("Recording stopped.")

# Train Model
elif menu_option == "Train Model":
    st.header("Train Random Forest Model")

    # Function to load data with consistent sequence length
    def load_data(data_path, sequence_length=30):
        sequences, labels = [], []
        actions = os.listdir(data_path)
        action_map = {action: idx for idx, action in enumerate(actions)}

        for action in actions:
            action_path = os.path.join(data_path, action)
            for file in os.listdir(action_path):
                if file.endswith(".npy"):
                    sequence = np.load(os.path.join(action_path, file))
                    if sequence.size == 0 or len(sequence.shape) < 2:  # Skip empty or malformed sequences
                        st.warning(f"Skipping malformed or empty sequence: {file}")
                        continue
                    if len(sequence) < sequence_length:
                        padding = np.zeros((sequence_length - len(sequence), sequence.shape[1]))
                        sequence = np.vstack([sequence, padding])
                    elif len(sequence) > sequence_length:
                        sequence = sequence[:sequence_length]
                    sequences.append(sequence)
                    labels.append(action_map[action])

        return np.array(sequences), np.array(labels), action_map

    # Button to start training
    train_button = st.button("Train Model")

    if train_button:
        with st.spinner("Loading data..."):
            X, y, action_map = load_data(data_dir)
            st.write(f"Loaded {len(X)} sequences for training.")
            num_actions = len(action_map)

        with st.spinner("Training the model..."):
            # Flatten the sequence data to make it suitable for Random Forest
            X_flat = X.reshape(X.shape[0], -1)

            # Train a Random Forest classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_flat, y)

            # Save the model
            model_path = os.path.join(data_dir, "rf_model.pkl")
            joblib.dump(model, model_path)

            st.success(f"Model trained and saved to: {model_path}")

# Use Model
elif menu_option == "Use Model":
    st.header("Use Trained Model")

    # Load the model
    model_path = os.path.join(data_dir, "rf_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success(f"Model loaded from: {model_path}")
    else:
        st.error("No trained model found. Please train the model first.")

    # Real-time gesture recognition
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Convert the uploaded file to a format that OpenCV can read
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        video = cv2.imdecode(file_bytes, 1)

        # Process the video here
        st.video(uploaded_file)
    else:
        # If no file is uploaded, use the webcam for live input
        st.write("Using webcam for live input...")

        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        if not cap.isOpened():
            st.error("Error: Could not access the webcam.")
        else:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame.")
                        break

                    # Flip the frame horizontally (optional)
                    frame = cv2.flip(frame, 1)

                    # Convert the frame to RGB (Streamlit uses RGB format)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the frame using MediaPipe
                    results = holistic.process(rgb_frame)

                    # Draw landmarks on frame
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    # Display the frame in Streamlit
                    frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()
