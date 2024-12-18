import logging
import queue
from pathlib import Path
import numpy as np
import streamlit as st
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from typing import NamedTuple

# Configuration
MODEL_PATH = "action.keras"
actions = ["blank", "Hello", "My", "Name", "Is"]
TARGET_FEATURE_DIM = 1662  # Expected feature dimension for the model

# Initialize Mediapipe for pose and hand detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Session-specific caching for model
@st.cache_resource
def load_sign_language_model():
    return load_model(MODEL_PATH)

model = load_sign_language_model()

# Helper functions
def extract_keypoints(results):
    """Extract keypoints from MediaPipe results."""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    keypoints = np.concatenate([pose, lh, rh])
    return np.pad(keypoints, (0, TARGET_FEATURE_DIM - len(keypoints)), mode="constant")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """Draw landmarks on the image."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Video frame callback for WebRTC
class Detection(NamedTuple):
    label: str
    score: float

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")

    # Process the frame through MediaPipe
    results = mediapipe_detection(image, mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5))[1]
    draw_styled_landmarks(image, results)

    # Extract keypoints from MediaPipe results
    keypoints = extract_keypoints(results)

    # Predict sign language action
    sequence = [keypoints]
    sequence = sequence[-30:]
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        confidence = res[np.argmax(res)]
        if confidence > 0.5:
            predicted_action = actions[np.argmax(res)]
            result_queue.put([Detection(label=predicted_action, score=confidence)])

    return frame

# WebRTC setup for video streaming
webrtc_ctx = webrtc_streamer(
    key="sign-language-recognition",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display predicted action
if st.checkbox("Show the detected action", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table([{"Label": detection.label, "Confidence": round(detection.score * 100, 2)} for detection in result])

# Streamlit UI for setting up the model and video
st.title("Sign Language Recognition with WebRTC")
st.markdown("This demo uses real-time video streaming with sign language recognition.")

st.markdown(
    "This demo is powered by a deep learning model for sign language recognition. "
    "It uses MediaPipe for pose and hand tracking, and a trained model for predicting sign language actions."
)

