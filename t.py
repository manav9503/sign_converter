import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from PIL import Image

# Load your pre-trained model (make sure to replace this with your actual model)
# For example, a model that takes in images and predicts sign language labels
model = tf.keras.models.load_model("C:\\Users\\manav\\Downloads\\action.h5")

# Mediapipe Hand Detection Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to process the hand gestures
def process_frame(frame):
    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # You can extract the landmarks to use for prediction
        hand_landmarks = results.multi_hand_landmarks[0]  # Assuming one hand
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        # Convert landmarks to numpy array (you can customize based on your model's requirement)
        landmarks = np.array(landmarks).flatten()

        # Model prediction (assuming the model expects flattened landmarks as input)
        prediction = model.predict(np.expand_dims(landmarks, axis=0))
        predicted_label = np.argmax(prediction)
        return frame, predicted_label
    return frame, None

# Streamlit WebApp
def main():
    st.title("Sign Language Recognition")

    # OpenCV Video Capture for live video feed
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process the frame
        frame, predicted_label = process_frame(frame)

        # Convert the frame to RGB format for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Show the processed frame
        stframe.image(img, channels="RGB", use_column_width=True)

        # Display the predicted label
        if predicted_label is not None:
            st.text(f"Predicted Sign: {predicted_label}")  # Replace with actual label names

    cap.release()

if __name__ == "__main__":
    main()
