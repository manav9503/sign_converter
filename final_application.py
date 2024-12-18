import os
import sys
import time
import streamlit as st
import cv2 as cv
from cv2 import aruco
import argparse
import numpy as np
import zipfile
from io import BytesIO
import sqlite3
import uuid
import json
import mediapipe as mp
from tensorflow.keras.models import load_model
import zipfile






# Admin code for validation
ADMIN_CODE = "12345"  # Replace with a secure code or environment variable

def validate_admin_code(input_code):
    """Validate the admin code."""
    return input_code == ADMIN_CODE



# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Load the model and define actions
MODEL_PATH = "action.keras"


model = load_model(MODEL_PATH)




actions = ["blank", "Hello", "My","Nmae","Is"]
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (117, 245, 16), (16, 117, 245)]
TARGET_FEATURE_DIM = 1662  # Expected feature dimension for the model

# SQLite Database for storing tokens and user data
def create_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()

    # Create the tokens table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            id TEXT PRIMARY KEY, 
            name TEXT, 
            email TEXT, 
            labels TEXT
        )
    ''')

    # Check if the 'labels' column exists and add it if necessary
    c.execute('''PRAGMA table_info(tokens)''')
    columns = [column[1] for column in c.fetchall()]
    if 'labels' not in columns:
        c.execute('''ALTER TABLE tokens ADD COLUMN labels TEXT''')

    # Create the training_data table if it doesn't exist
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



def get_all_users():
    """Get all users and their details."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()
    c.execute("SELECT * FROM tokens")
    users = c.fetchall()
    conn.close()
    return users

# Function to fetch all distinct labels
def get_all_labels():
    """Fetch distinct labels from the training_data table."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT label FROM training_data")
    labels = [row[0] for row in c.fetchall()]
    conn.close()
    return labels

# Function to fetch training data for a specific label
def get_training_data_by_label(label):
    """Fetch all training data for a specific label."""
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()
    c.execute("""
        SELECT sequence, frame, keypoints 
        FROM training_data 
        WHERE label = ?
    """, (label,))
    data = c.fetchall()
    conn.close()
    return data

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

def create_zip_from_directory(directory_path):
    """Create a zip file from the directory."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory_path))
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit app
st.set_page_config(page_title="Sign Language Recognition", page_icon="🤖", layout="wide")

st.sidebar.title("Menu")
menu = st.sidebar.radio("Select Mode", ["Real-Time Detection", "Collect Training Data", "View Labels","Admin Panel"])

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

if menu == "Real-Time Detection":
    st.title("Real-Time Action Detection 🤖")
    st.markdown("<hr>", unsafe_allow_html=True)
    start_button = st.button("Start Detection", key="start_detection")
    stop_button = st.button("Stop Detection", key="stop_detection")
    predicted_labels = st.empty()

    # Variables
    sequence = []
    sentence = []
    predictions = []

    if start_button:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error accessing the webcam. Please ensure it is connected and try again.")
        else:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

            prev_time = 0
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                stframe = st.empty()

                recording = True  # Variable to control recording state

                while cap.isOpened() and recording:
                    ret, frame = cap.read()
                    if not ret:
                        st.write("Error accessing the webcam.")
                        break

                    if stop_button:  # Check if Stop Recording button is pressed
                        recording = False  # Stop recording
                        st.write("Detection stopped.")
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        try:
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            confidence = res[np.argmax(res)]

                            predictions.append(np.argmax(res))

                            if confidence > 0.5:
                                predicted_action = actions[np.argmax(res)]
                                if len(sentence) == 0 or predicted_action != sentence[-1]:
                                    sentence.append(predicted_action)

                            if len(sentence) > 5:
                                sentence = sentence[-5:]

                            image = prob_viz(res, actions, image, colors)

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

                    predicted_labels.text(f"Predicted Action: {' '.join(sentence)}")
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time
                    st.text(f"FPS: {fps:.2f}")
                    stframe.image(image, channels="BGR")

            cap.release()
            cv.destroyAllWindows()

elif menu == "Collect Training Data":
    st.title("Collect Training Data 📷")
    token = st.text_input("Enter your token:")
    if token:
        user = get_user_from_token(token)  # Replace with actual user lookup
        if user:
            st.write(f"Welcome {user[1]}! You are authorized to record data.")
            label_name = st.text_input("Enter the label name:")
            start_recording = st.button("Start Recording")
            stop_recording = st.button("Stop Recording")
            
            if start_recording and label_name:
                st.session_state.recording = True
                st.session_state.stop = False  # Reset stop flag
                cap = cv2.VideoCapture(0)
                stframe = st.empty()

                try:
                    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        for sequence in range(30):  # Number of sequences
                            if not st.session_state.recording:  # Check if Stop button was pressed
                                break

                            for frame_num in range(30):  # Frames per sequence
                                if not st.session_state.recording:  # Check if Stop button was pressed
                                    break
                                
                                ret, frame = cap.read()
                                if not ret:
                                    st.error("Error accessing the webcam.")
                                    break
                                
                                image, results = mediapipe_detection(frame, holistic)
                                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                                keypoints = extract_keypoints(results)

                                # Save data to the database
                                save_training_data_to_db(token, label_name, sequence, frame_num, keypoints)

                                # Display the current frame on Streamlit
                                stframe.image(image, channels="BGR")
                                
                                if stop_recording:  # Stop button logic
                                    st.session_state.recording = False
                                    st.session_state.stop = True
                                    st.warning("Recording stopped.")
                                    break

                except Exception as e:
                    st.error(f"An error occurred: {e}")

                finally:
                    # Ensure the camera is released and all windows are closed
                    if cap.isOpened():
                        cap.release()
                    cv.destroyAllWindows()

                if st.session_state.recording:
                    st.success("Data collection completed!")
                else:
                    st.warning("Data collection was stopped.")
        else:
            st.error("Invalid token. Please create a new token below.")
            name = st.text_input("Enter your name:")
            email = st.text_input("Enter your email:")

            if st.button("Create Token"):
                if name and email:
                    token = str(uuid.uuid4())  # Generate a unique token
                    save_token_to_db(token, name, email, "")
                    st.success(f"Authenticated successfully! Your token: {token}")
                else:
                    st.error("Please enter both name and email to generate a token.")


elif menu == "View Labels":
    st.title("View Labels 📂")
    conn = sqlite3.connect('tokens.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT label FROM training_data")
    labels = [row[0] for row in c.fetchall()]
    conn.close()

    if labels:
        st.write("### Available Labels")
        for label in labels:
            st.write(f"- {label}")
    else:
        st.warning("No labels found. Collect training data first!")


     


elif menu == "Admin Panel":
    st.title("Admin Panel 🔐")

    # Initialize session state for login tracking
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    # Admin login form
    if not st.session_state.admin_logged_in:
        admin_password = st.text_input("Enter Admin Password:", type="password")

        if st.button("Login"):
            if validate_admin_code(admin_password):
                st.session_state.admin_logged_in = True
                st.success("Access Granted! Welcome Admin.")
            else:
                st.error("Incorrect password. Please try again.")

    # If logged in, display admin panel content
    if st.session_state.admin_logged_in:
        # Display all user data
        st.subheader("All Users and Their Labels")
        users = get_all_users()
        if users:
            # Include tokens in the displayed data
            user_data = [{"Name": user[1], "Email": user[2]} for user in users]
            st.table(user_data)
        else:
            st.warning("No user data found.")

        # Option to download data for each label individually
        st.subheader("Download Training Data by Label")

        # Fetch all distinct labels
        labels = get_all_labels()
        i = 0
        if labels:
            for label in labels:
                st.markdown(f"### {i} {label} Data")
                i += 1

                # Fetch training data for the label
                data = get_training_data_by_label(label)

                if data:
                    # Format data into JSON
                    label_data = [
                        {"sequence": d[0], "frame": d[1], "keypoints": json.loads(d[2])}
                        for d in data
                    ]

                    # Create a ZIP file in memory
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        zip_file.writestr(f"{label}_data.json", json.dumps(label_data, indent=4))

                    zip_buffer.seek(0)

                    # Add download button for the ZIP file
                    st.download_button(
                        label=f"Download {label} Training Data as ZIP",
                        data=zip_buffer,
                        file_name=f"{label}_training_data.zip",
                        mime="application/zip"
                    )
                else:
                    st.warning(f"No training data found for label: {label}")
        else:
            st.warning("No labels found. Please collect training data first!")

