import streamlit as st
import pandas as pd


st.title('Emotion Classification')

text = st.text_area('Enter your text:', '')

if st.button('Classify'):

    st.write(f'The emotion detected is: ....')

# Evaluation metrics
st.subheader('Model Evaluation')

st.write(f'Accuracy:...')