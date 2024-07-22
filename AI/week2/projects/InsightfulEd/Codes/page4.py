import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 



def page4():
    st.title("Student Performance Prediction")

    # Load and preprocess the dataset
    df = pd.read_csv(r"C:/Users/simo2/Desktop/EVC_Training/Week_2/project2/student-por.csv")

    # Drop unnecessary columns
    dropped_columns = ['school','address', 'famsize','Pstatus','reason', 'guardian']
    df = df.drop(dropped_columns, axis=1)

    # Encode categorical variables
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'F' else 0)
    jobs = ['at_home', 'health', 'other', 'services', 'teacher']
    for job in jobs:
        df[f"Mjob_{job}"] = df['Mjob'].apply(lambda x: 1 if job == x else 0)
        df[f"Fjob_{job}"] = df['Fjob'].apply(lambda x: 1 if job == x else 0)
    df = df.drop(['Fjob', 'Mjob'], axis=1)
    binary_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for column in binary_columns:
        df[column] = df[column].apply(lambda x: 1 if x == 'yes' else 0)

    # Select relevant features
    df_clean = df[['sex', 'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                   'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
                   'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
                   'health', 'absences', 'Mjob_at_home', 'Fjob_at_home',
                   'Mjob_health', 'Fjob_health', 'Mjob_other', 'Fjob_other', 'Mjob_services',
                   'Fjob_services', 'Mjob_teacher', 'Fjob_teacher', 'G3']]

    X = df_clean[df_clean.columns[:-1]]
    y = df_clean['G3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Streamlit UI
    st.write("Please provide the following details to predict your grade:")

    # Input fields for features
    sex = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=15, max_value=22, value=17)
    Medu = st.slider("Mother's education (0-4)", min_value=0, max_value=4, value=2)
    Fedu = st.slider("Father's education (0-4)", min_value=0, max_value=4, value=2)
    traveltime = st.slider("Travel time to school (1-4)", min_value=1, max_value=4, value=1)
    studytime = st.slider("Weekly study time (1-4)", min_value=1, max_value=4, value=2)
    failures = st.slider("Past class failures (0-4)", min_value=0, max_value=4, value=0)
    schoolsup = st.selectbox("School support", ["Yes", "No"])
    famsup = st.selectbox("Family support", ["Yes", "No"])
    paid = st.selectbox("Extra paid classes", ["Yes", "No"])
    activities = st.selectbox("Extra-curricular activities", ["Yes", "No"])
    nursery = st.selectbox("Attended nursery school", ["Yes", "No"])
    higher = st.selectbox("Wants to pursue higher education", ["Yes", "No"])
    internet = st.selectbox("Access to internet at home", ["Yes", "No"])
    romantic = st.selectbox("In a romantic relationship", ["Yes", "No"])
    famrel = st.slider("Quality of family relationships (1-5)", min_value=1, max_value=5, value=4)
    freetime = st.slider("Free time after school (1-5)", min_value=1, max_value=5, value=3)
    goout = st.slider("Going out with friends (1-5)", min_value=1, max_value=5, value=3)
    Dalc = st.slider("Workday alcohol consumption (1-5)", min_value=1, max_value=5, value=1)
    Walc = st.slider("Weekend alcohol consumption (1-5)", min_value=1, max_value=5, value=2)
    health = st.slider("Current health status (1-5)", min_value=1, max_value=5, value=4)
    absences = st.slider("Number of school absences", min_value=0, max_value=93, value=4)
    Mjob = st.selectbox("Mother's job", jobs)
    Fjob = st.selectbox("Father's job", jobs)

    # Convert input to numerical values
    input_data = {
        'sex': 1 if sex == "Female" else 0,
        'age': age,
        'Medu': Medu,
        'Fedu': Fedu,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': 1 if schoolsup == "Yes" else 0,
        'famsup': 1 if famsup == "Yes" else 0,
        'paid': 1 if paid == "Yes" else 0,
        'activities': 1 if activities == "Yes" else 0,
        'nursery': 1 if nursery == "Yes" else 0,
        'higher': 1 if higher == "Yes" else 0,
        'internet': 1 if internet == "Yes" else 0,
        'romantic': 1 if romantic == "Yes" else 0,
        'famrel': famrel,
        'freetime': freetime,
        'goout': goout,
        'Dalc': Dalc,
        'Walc': Walc,
        'health': health,
        'absences': absences,
        'Mjob_at_home': 1 if Mjob == "at_home" else 0,
        'Fjob_at_home': 1 if Fjob == "at_home" else 0,
        'Mjob_health': 1 if Mjob == "health" else 0,
        'Fjob_health': 1 if Fjob == "health" else 0,
        'Mjob_other': 1 if Mjob == "other" else 0,
        'Fjob_other': 1 if Fjob == "other" else 0,
        'Mjob_services': 1 if Mjob == "services" else 0,
        'Fjob_services': 1 if Fjob == "services" else 0,
        'Mjob_teacher': 1 if Mjob == "teacher" else 0,
        'Fjob_teacher': 1 if Fjob == "teacher" else 0
    }

    input_df = pd.DataFrame([input_data])

    # Make prediction
    if st.button("Predict", use_container_width=True):
        prediction = lr.predict(input_df)[0]

        st.write(f"Predicted final grade: {prediction:.2f} out of 20")

page4()
