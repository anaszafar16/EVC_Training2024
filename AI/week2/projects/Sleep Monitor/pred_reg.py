import streamlit as st
import joblib
import pandas as pd

# Define grade class mapping
grade_class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

@st.cache(allow_output_mutation=True)
def load_model():
    # Load your trained model
    model = joblib.load("C:/Users/PC/Desktop/project2/reg_model.joblib")
    return model

def predict_sleep_duration(quality_of_sleep, physical_activity, bmi_category, sleep_condition, age, gender, study_time, absences, parental_support, grade_class, model):
    # Encode categorical variables
    gender_encoded = 0 if gender == "Male" else 1
    grade_class_encoded = list(grade_class_mapping.keys())[list(grade_class_mapping.values()).index(grade_class)]
    sleep_condition_encoded = 0 if sleep_condition == "Sleep Disorder healthy" else (2 if sleep_condition == "Sleep Apnea" else 1)
    bmi_category_encoded = 3 if bmi_category == "Overweight" else (2 if bmi_category == "Obese" else (1 if bmi_category == "Normal Weight" else 0))

    # Convert input values to dataframe
    data = pd.DataFrame({
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity],
        'BMI Category': [bmi_category_encoded],
        'Sleep Disorder': [sleep_condition_encoded],
        'Age': [age],
        'Gender': [gender_encoded],
        'StudyTimeWeekly': [study_time],
        'Absences': [absences],
        'ParentalSupport': [parental_support],
        'GradeClass': [grade_class_encoded]
    })

    # Make prediction using the model
    prediction = model.predict(data)[0]

    # Return the predicted sleep duration
    return prediction

# Create the Streamlit web app
def main():
    st.title("Sleep Duration Prediction")
    st.write("Enter the following information to predict sleep duration:")

    # Load the model
    model = load_model()

    # Get user input
    quality_of_sleep = st.slider("Quality of Sleep", 1, 10)
    physical_activity = st.slider("Physical Activity Level (mins)", 30, 90)
    bmi_category = st.selectbox("BMI Category", ["Overweight", "Normal", "Obese", "Normal Weight"])
    sleep_condition = st.selectbox("Sleep Disorder", ["Sleep Disorder healthy", "Sleep Apnea", "Insomnia"])
    age = st.number_input("Age", min_value=1, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    study_time = st.number_input("Study Time (Weekly)", min_value=0, max_value=20, value=10)
    absences = st.number_input("Absences", min_value=0, max_value=30, value=0)
    parental_support = st.slider("Parental Support : 0--None ,1--Low ,2--Moderate ,3--High ,4--Very High", 0, 4)
    grade_class = st.selectbox("Grade Class", ["A", "B", "C","D","F"])

    if st.button("Predict Sleep Duration"):
        # Perform prediction
        sleep_duration = predict_sleep_duration(quality_of_sleep, physical_activity, bmi_category, sleep_condition, age, gender, study_time, absences, parental_support, grade_class, model)

        # Display result
        st.write("Predicted Sleep Duration:", round(sleep_duration, 2), "hours")


# Run the app
if __name__ == "__main__":
    main()
    