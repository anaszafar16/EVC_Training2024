# Sign Language Interpretation 

This repository contains a deep learning model based on YOLO (You Only Look Once) for detecting sign language gestures and interpreting them into voice. The project aims to assist individuals who cannot speak by providing a seamless way to communicate using sign language. By leveraging advanced computer vision techniques and speech synthesis technologies, this system translates hand signs into spoken words, enhancing accessibility and communication for those with speech impairments.



## Models

### Deep Learning Model

- **Model Name:** YOLOv8
- **Purpose:** Sign Language Detection and Interpretation
- **Details:** The YOLOv8 model detects and interprets sign language gestures in real-time, translating them into spoken words. This model leverages the efficiency and accuracy of the YOLO architecture to identify hand signs from video input and convert them into corresponding voice outputs, providing an effective communication tool for individuals with speech impairments.
- **Special Thanks To**   [@Daniamhz](https://github.com/Daniamhz) and [@Janaalharbii](https://github.com/Janaalharbii)

### Text-to-Speech Model

- **Model Name:** pyttsx3
- **Purpose:** Voice Synthesis
- **Details:** The pyttsx3 library is used to convert the detected sign language gestures into spoken words. It provides an offline text-to-speech conversion, ensuring that the interpreted signs are instant
- **Special Thanks To**   [@Hassan](https://github.com/hs-kln)

## Data

### Sign Language Detection Data

- **Data Name:** Custom Sign Language Dataset
- **Source:** Created by our team and hosted on Roboflow for public use
- **Details:** This dataset consists of annotated images and videos of various sign language gestures, curated and labeled by our team to train the YOLOv8 model.
