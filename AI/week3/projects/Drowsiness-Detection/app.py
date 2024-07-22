import streamlit as st
import cv2
from ultralytics import YOLO
import cvzone
import math
import pygame
import numpy as np

# Initialize pygame mixer
pygame.mixer.init()

# Load sound
alert_sound = pygame.mixer.Sound('alarm.mp3')

# Load the model
model = YOLO('best.pt')

# Reading the classes
classnames = ['Drowsy', 'Awake']

# Streamlit UI
st.set_page_config(layout="wide")  # Set wide layout

# Add the logo to the sidebar
logo_path = "logo.jpg"  # Use the uploaded file path
st.sidebar.empty()  # Add empty space
st.sidebar.image(logo_path, use_column_width=True)

# Create a sidebar for navigation
st.sidebar.title("Options")
page = st.sidebar.selectbox("Choose a page", ["Webcam Detection", "Image Upload"])

st.title("Drowsiness Detection")

if page == "Webcam Detection":
    st.header("Real-Time Drowsiness Detection")

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        start_button = st.button('Start Webcam')

    with col2:
        stop_button = st.button('Stop Webcam')

    alert_placeholder = st.empty()  # Placeholder for alerts
    stframe = st.empty()
    status_text = st.empty()
    message_text = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        drowsy_count = 0  # Counter for consecutive "Drowsy" detections
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_text.write("Failed to grab frame")
                break
            
            frame = cv2.resize(frame, (640, 480))

            # Run the model on the frame
            result = model(frame, stream=True)

            # Flag to track if "Drowsy" is detected in this frame
            drowsy_detected = False

            # Getting bbox, confidence, and class name information to work with
            for info in result:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    if confidence > 50:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                           scale=1.5, thickness=2)
                        if classnames[Class] == 'Drowsy':
                            drowsy_detected = True

            # Increment the counter if "Drowsy" is detected, otherwise reset the counter
            if drowsy_detected:
                drowsy_count += 1
                status_text.write("Drowsiness detected!")
            else:
                drowsy_count = 0
                status_text.write("Monitoring...")

            # Play alert sound and send message if "Drowsy" is detected 3 or more times
            if drowsy_count >= 3:
                pygame.mixer.Sound.play(alert_sound)
                alert_placeholder.markdown(
                    f'<div style="color: red; font-size: 24px; border: 2px solid red; padding: 10px;">**Be careful! Drowsiness detected!**</div>',
                    unsafe_allow_html=True,
                )
                drowsy_count = 0  # Reset the counter after playing the sound

            # Convert image back to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the image
            stframe.image(frame, channels="RGB")

            # Check if stop button is pressed
            if stop_button:
                break

        cap.release()
        status_text.write("Webcam stopped.")
        message_text.write("")
        alert_placeholder.empty()

elif page == "Image Upload":
    st.header("Drowsiness Detection on Image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Perform prediction
        results = model(frame, stream=True)

        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cv2.putText(frame, f'{classnames[Class]} {confidence}%', (x1 + 8, y1 + 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert image back to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the image
        st.image(frame, channels="RGB")

