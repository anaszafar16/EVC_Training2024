import cv2
import math
import streamlit as st
from ultralytics import YOLO
import pandas as pd 

# Initialize model and virtual receipt
model = YOLO('yolov8n.pt')
virtual_receipt = {}
total = 0
# Category dictionary
category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Pricing function
def price_func(label):
    if label == 'cell phone':
        return 4000
    elif label == 'cup':
        return 5
    elif label == 'remote':
        return 15
    elif label == 'laptop':
        return 3000
    elif label == 'tv':
        return 1500
    elif label == 'chair':
        return 200
    elif label == 'bottle':
        return 2
    else:
        return 20

# Streamlit UI
st.title("Virtual Checkout System Prototype")
st.subheader("Using YOLOv8 for Object Detection")
st.write('Welcome to Sunshine market! Please place your items in the frame to be added to your receipt.')
if 'run' not in st.session_state:
    st.session_state.run = False

# Function to start webcam
def start_webcam():
    st.session_state.run = True

# Function to stop webcam
def stop_webcam():
    st.session_state.run = False
    st.write("**VIRTUAL RECEIPT**")

    for key, value in virtual_receipt.items():
        st.write(key, ":   ", value)
    st.write("**Total:** ", sum(virtual_receipt.values()), " SAR")
    st.write("Thank you for shopping at Sunshine Market! See you again soon!")

# Start and Stop buttons
start_button = st.button('Start', on_click=start_webcam)
stop_button = st.button('Stop', on_click=stop_webcam)

# Placeholder for webcam feed
frame_placeholder = st.empty()

# Placeholder for virtual receipt
receipt_placeholder = st.empty()
total_placeholder = st.empty()
# Webcam loop
cap = cv2.VideoCapture(0)

while st.session_state.run:
# Initialize video capture
    # Stream video frames in real-time
    ret, frame = cap.read()
    if not ret:
        st.error('Cannot receive frame. Exiting...')
        break

    # Model prediction
    results = model(frame)
    for info in results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            label = category_dict[Class]
            price = price_func(label)
            if label != 'person':
                if confidence > 80:
                    # Rectangle and text on webcam
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cv2.putText(frame, f'{label}: {confidence:.2f}% confidence', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f'Price: {price} SAR', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                    # Update virtual receipt
                    virtual_receipt[label] = price
                    
        
    # Convert frame to RGB format for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # Display frame in Streamlit
    frame_placeholder.image(frame, channels='RGB')

    # Display virtual receipt
    receipt_placeholder.write(virtual_receipt)
cap.release()
cv2.destroyAllWindows()


