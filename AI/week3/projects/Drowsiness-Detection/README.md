# Drowsiness Detection System

This project aims to develop a real-time drowsiness detection system using a pre-trained YOLOv8s model. The system can detect drowsiness in individuals through webcam footage or uploaded images, providing alerts to prevent potential accidents.

## Project Description

Drowsiness while driving or operating heavy machinery can lead to dangerous situations. This project leverages machine learning and computer vision to identify signs of drowsiness and alert the user. The system can be used in various applications, such as automotive safety, workplace safety, and general health monitoring.

## Goals

1. **Real-Time Drowsiness Detection:** Use a webcam to detect signs of drowsiness in real-time.
2. **Image Upload Detection:** Allow users to upload images for drowsiness detection.
3. **Alert System:** Provide audible and visual alerts when drowsiness is detected.
4. **User-Friendly Interface:** Create an intuitive interface using Streamlit for easy interaction.

## Setup Instructions

### Prerequisites

Ensure you have the following libraries installed:

- `streamlit`
- `opencv-python`
- `ultralytics`
- `cvzone`
- `pygame`
- `numpy`

### Usage
## Webcam Detection
- Select the "Webcam Detection" option from the sidebar.
- Click the "Start Webcam" button to begin real-time detection.
- If drowsiness is detected 3 or  more  times, an alert will be triggered.
- Click the "Stop Webcam" button to stop detection.
## Image Upload Detection
- Select the "Image Upload" option from the sidebar.
- Upload an image file.
- The system will process the image and display the detection results.
## Project Structure
- app.py: Main application file containing the Streamlit code.
- best.pt: Pre-trained YOLO model for drowsiness detection.
- alarm.mp3: Audio file for alert sound.
- logo.jpg: Logo image for the application sidebar.




### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/drowsiness-detection.git
    cd drowsiness-detection
    ```

2. Install the required libraries:
    ```bash
    pip install streamlit opencv-python ultralytics cvzone pygame numpy
    ```

3. Download the pre-trained YOLO model `best.pt` and place it in the project directory.

### Running the Application

To run the application, use the following command:
```bash
streamlit run app.py
```
### Demo
![demo](https://github.com/user-attachments/assets/d1cad65e-8960-48c2-b49f-1493057bb97f)
