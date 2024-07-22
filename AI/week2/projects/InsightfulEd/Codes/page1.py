# page1.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def page1():
    # Load the dataset
    df = pd.read_csv(r"C:\Users\simo2\Desktop\EVC_Training\Week_2\project2\dataset.csv")

    # Define the selected features
    best_features = [
        'Scholarship holder',
        'Gender',
        'Age at enrollment',
        "Mother's occupation",
        "Father's occupation",
        'Debtor',
        'Marital status'
    ]

    # Extract features and target variable
    X = df[best_features]
    y = df['Target']

    # Encode the target variable
    y = y.astype('category').cat.codes

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the model
    model = RandomForestClassifier(random_state=42)

    # Define a simpler parameter grid for hyperparameter tuning to speed up the process
    param_dist = {
        'n_estimators': [50, 100],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform RandomizedSearchCV with fewer iterations for faster processing
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=2)
    random_search.fit(X_train, y_train)

    # Get the best estimator
    best_model = random_search.best_estimator_

    # Analyze the dataset to find the most and least common occupations
    mother_occ_counts = df["Mother's occupation"].value_counts()
    father_occ_counts = df["Father's occupation"].value_counts()

    # Define high, middle, and low occupation categories for mother
    mother_high_occ = mother_occ_counts.head(3).index.tolist()
    mother_low_occ = mother_occ_counts.tail(3).index.tolist()
    mother_middle_occ = mother_occ_counts[~mother_occ_counts.index.isin(mother_high_occ + mother_low_occ)].index.tolist()

    # Define high, middle, and low occupation categories for father
    father_high_occ = father_occ_counts.head(3).index.tolist()
    father_low_occ = father_occ_counts.tail(3).index.tolist()
    father_middle_occ = father_occ_counts[~father_occ_counts.index.isin(father_high_occ + father_low_occ)].index.tolist()

    # Convert all marital status values to strings
    df['Marital status'] = df['Marital status'].astype(str)
    marital_status_counts = df['Marital status'].value_counts()

    # Define a function to safely get the most common status
    def get_common_status(status_list, keyword):
        for status in status_list:
            if keyword.lower() in status.lower():
                return status
        return None

    # Define marital statuses
    single = get_common_status(marital_status_counts.index, 'single')
    married = get_common_status(marital_status_counts.index, 'married')
    divorced = get_common_status(marital_status_counts.index, 'divorced')
    widowed = get_common_status(marital_status_counts.index, 'widowed')

    # Ensure all statuses were found, else assign a default value
    if not single: single = 'Single'
    if not married: married = 'Married'
    if not divorced: divorced = 'Divorced'
    if not widowed: widowed = 'Widowed'

    # Streamlit app
    st.title("Student Dropout Prediction")

    st.write("Please provide the following details:")

    # Get user input
    scholarship_holder = st.selectbox('Scholarship holder', ['Yes', 'No'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age_at_enrollment = st.number_input('Age at enrollment', min_value=15, max_value=50, value=20)
    mothers_occupation = st.selectbox("Mother's occupation", ['High Occupation', 'Middle Occupation', 'Low Occupation'])
    fathers_occupation = st.selectbox("Father's occupation", ['High Occupation', 'Middle Occupation', 'Low Occupation'])
    debtor = st.selectbox('Debtor', ['Yes', 'No'])
    marital_status = st.selectbox('Marital status', ['Single', 'Married', 'Divorced', 'Widowed'])

    # Encode user input
    def encode_occupation(occupation, high_occ, middle_occ, low_occ):
        if occupation == 'High Occupation':
            return high_occ[0]  # Using the first occupation as a representative
        elif occupation == 'Middle Occupation':
            return middle_occ[0]  # Using the first occupation as a representative
        else:
            return low_occ[0]  # Using the first occupation as a representative

    def encode_marital_status(status):
        if status == 'Single':
            return single
        elif status == 'Married':
            return married
        elif status == 'Divorced':
            return divorced
        else:
            return widowed

    input_data = pd.DataFrame({
        'Scholarship holder': [1 if scholarship_holder == 'Yes' else 0],
        'Gender': [1 if gender == 'Male' else 0],
        'Age at enrollment': [age_at_enrollment],
        "Mother's occupation": [encode_occupation(mothers_occupation, mother_high_occ, mother_middle_occ, mother_low_occ)],
        "Father's occupation": [encode_occupation(fathers_occupation, father_high_occ, father_middle_occ, father_low_occ)],
        'Debtor': [1 if debtor == 'Yes' else 0],
        'Marital status': [encode_marital_status(marital_status)]
    })

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Standardize user input
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = best_model.predict(input_data)
    prediction_proba = best_model.predict_proba(input_data)

    # Display prediction
    probability = prediction_proba[0][prediction[0]] * 100
    result = "Dropout" if probability >= 80 else "Not Dropout"
    st.write(f"Prediction: {result}")
    st.write(f"Prediction Probability: {probability:.2f}%")

    if st.button('Show Feature Importance'):
        feature_importances = best_model.feature_importances_
        features = pd.Series(feature_importances, index=X.columns)
        st.write(features.sort_values(ascending=False))

if __name__ == '__main__':
    page1()
