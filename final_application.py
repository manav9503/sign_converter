import os
import sys
import time
import streamlit as st
import numpy as np
import sqlite3
import uuid
import json
import mediapipe as mp
from tensorflow.keras.models import load_model
import zipfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Admin code for validation
ADMIN_CODE = "12345"  # Replace with a secure code or environment variable

def validate_admin_code(input_code):
    """Validate the admin code."""
    return input_code == ADMIN_CODE

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Load the model and define actions
MODEL_PATH = "C:/Users/manav/OneDrive/Desktop/action2.keras"
model = load_model(MODEL_PATH)

actions = ["blank", "Hello", "My", "Name", "Is"]
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (117, 245, 16), (16, 117, 245)]
TARGET_FEATURE_DIM = 1662  # Expected feature dimension for the model

# SQLite Database for storing tokens and user data
def create_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()

    # Check if 'tokens' table exists and add 'labels' column if not present
    c.execute('''PRAGMA table_info(tokens)''')
    columns = [column[1] for column in c.fetchall()]
    if 'labels' not in columns:
        c.execute('''ALTER TABLE tokens ADD COLUMN labels TEXT''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            id TEXT PRIMARY KEY, 
            name TEXT, 
            email TEXT, 
            labels TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT,
            label TEXT,
            sequence INTEGER,
            frame INTEGER,
            keypoints TEXT,
            FOREIGN KEY(token) REFERENCES tokens(id)
        )
    ''')
    conn.commit()
    conn.close()

def save_token_to_db(token, name, email, labels):
    """Save token and user details to the database."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()
    c.execute("INSERT INTO tokens (id, name, email, labels) VALUES (?, ?, ?, ?)", (token, name, email, labels))
    conn.commit()
    conn.close()

def get_user_from_token(token):
    """Get user details from the token."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()
    c.execute("SELECT * FROM tokens WHERE id=?", (token,))
    user = c.fetchone()
    conn.close()
    return user

def get_all_users():
    """Get all users and their details."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()
    c.execute("SELECT * FROM tokens")
    users = c.fetchall()
    conn.close()
    return users

# Initialize the database
create_db()

# Helper functions
def prob_viz(res, actions, input_frame, colors):
    """Visualize the probabilities (debugging purposes)."""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(
            output_frame,
            f"{actions[num]}: {prob:.2f}",
            (5, 80 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output_frame

if "recording" not in st.session_state:
    st.session_state.recording = False
if "stop" not in st.session_state:
    st.session_state.stop = False

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results."""
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    keypoints = np.concatenate([pose, lh, rh])
    return np.pad(keypoints, (0, TARGET_FEATURE_DIM - len(keypoints)), mode="constant")

def save_training_data_to_db(token, label, sequence, frame, keypoints):
    """Save collected keypoints data to the database."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()
    c.execute("""
        INSERT INTO training_data (token, label, sequence, frame, keypoints)
        VALUES (?, ?, ?, ?, ?)
    """, (token, label, sequence, frame, json.dumps(keypoints.tolist())))
    conn.commit()
    conn.close()

# Streamlit app
st.set_page_config(page_title="Sign Language Recognition", page_icon="🤖", layout="wide")

st.sidebar.title("Menu")
menu = st.sidebar.radio("Select Mode", ["Real-Time Detection", "Collect Training Data", "View Labels", "Admin Panel"])

# Styling the UI
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #008CBA;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #005f73;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 10px;
    }
    .stTextInput>div>div>label {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit WebRTC Video Transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.predictions = []

    def transform(self, frame):
        # Get the image as an array
        image = frame.to_ndarray(format="bgr24")

        # Process the frame using MediaPipe
        image, results = mediapipe_detection(image, mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5))
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) == 30:
            try:
                res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                confidence = res[np.argmax(res)]

                self.predictions.append(np.argmax(res))

                if confidence > 0.5:
                    predicted_action = actions[np.argmax(res)]
                    if len(self.sentence) == 0 or predicted_action != self.sentence[-1]:
                        self.sentence.append(predicted_action)

                if len(self.sentence) > 5:
                    self.sentence = self.sentence[-5:]

                image = prob_viz(res, actions, image, colors)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        # Display the predicted action
        st.text(f"Predicted Action: {' '.join(self.sentence)}")
        return image

# Streamlit app logic
if menu == "Real-Time Detection":
    st.title("Real-Time Action Detection 🤖")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)
