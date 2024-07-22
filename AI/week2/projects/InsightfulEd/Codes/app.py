import streamlit as st
from streamlit_option_menu import option_menu

# Import individual page scripts
from page1 import page1
from page2 import page2
from page3 import page3
from page4 import page4

st.set_page_config(page_title="Multi-Page App", layout="wide")

# Create sidebar menu
with st.sidebar:
    
          
    image_path = "C:/Users/simo2/Desktop/EVC_Training/Week_2/project2/holographic.png"  # Replace with your image format (jpg, jpeg, png)
    title_path = "C:/Users/simo2/Desktop/EVC_Training/Week_2/project2/title.png"  # Replace with your image format (jpg, jpeg, png)

    # Display the image
    st.image(image_path, width=300)  # Adjust width as needed
    st.image(title_path, width=290)  # Adjust width as needed
    
    selected = option_menu(
        menu_title="Main Menu",
        options=["Student Dropout Prediction", "Bullying Prediction", "Income Prediction", "Student GPA Prediction"],
        icons=["house", "gear", "bar-chart-line", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )

# Load selected page
if selected == "Student Dropout Prediction":
    page1()
elif selected == "Bullying Prediction":
    page2()
elif selected == "Income Prediction":
    page3()
elif selected == "Student GPA Prediction":
    page4()



