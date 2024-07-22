import streamlit as st

st.title('Emotion Classification')


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.image("ea.jpeg")
st.write("")
st.write("Classifying...")
st.write("Predictions:...")
st.write("Predicted class: Happy ")
