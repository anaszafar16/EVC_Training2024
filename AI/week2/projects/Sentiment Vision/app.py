import pandas as pd
import numpy 
import streamlit as st

st.title('Emotion Classification')

st.image("logo.png")


col1 , col2 = st.columns([2,2])

with col1:
    if st.button("Text Sentiment Analysis"):
        st.switch_page("pages/text.py")


with col2:
    if st.button("Facial Emotion Analysis"):
        st.switch_page("pages/img.py")