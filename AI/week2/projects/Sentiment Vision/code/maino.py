import streamlit as st

st.title('Emotion Classification')

st.image("C:\\Users\\abdul\\Downloads\\passport-control-icon-linear-vector-illustration-airport-travel-collection-outline-thin-line-symbol-use-web-mobile-187603110.png")


col1 , col2 = st.columns([2,2])

with col1:
    if st.button("Text Sentiment Analysis"):
        st.switch_page("pages\\app1.py")


with col2:

    if st.button("Facial Emotion Analysis"):
        st.switch_page("pages\\vision.py")