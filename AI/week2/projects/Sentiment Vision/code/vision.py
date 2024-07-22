import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('C:\\Users\\abdul\\Downloads\\my_model.h5')

def preprocess_image(image):
    # Resize the image to 48x48 pixels
    img = image.resize((48, 48))
    # Convert the image to grayscale
    img = img.convert('L')
    # Normalize the image
    img = np.array(img) / 255.0
    # Expand dimensions to match the model's input shape
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

def predict_emotion(image, model, categories):
    # Preprocess the image
    img = preprocess_image(image)
    # Make the prediction
    prediction = model.predict(img)
    # Get the index of the highest probability class
    predicted_class = np.argmax(prediction)
    # Map the index to the class label
    predicted_label = categories[predicted_class]
    return predicted_label

# Streamlit app
st.title("Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predictions = predict_emotion(image, model, categories=['surprise', 'Angry', 'Happy', 'Neutral', 'Sad', 'surprise'])
    st.write("Predicted class:", predictions)