import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

def page2():
    st.title("Bullying Prediction")

    # Load and preprocess the dataset
    df = pd.read_csv(r"C:\Users\simo2\Desktop\EVC_Training\Week_2\project2\Bullying_2018.csv", sep=';')
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.dropna(how='any', inplace=True)
    df.drop(['Bullied_not_on_school_property_in_past_12_months'], axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df['Bullied_on_school_property_in_past_12_months'] = le.fit_transform(df['Bullied_on_school_property_in_past_12_months'])
    df['Cyber_bullied_in_past_12_months'] = le.fit_transform(df['Cyber_bullied_in_past_12_months'])
    df['Custom_Age'] = le.fit_transform(df['Custom_Age'])
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Physically_attacked'] = le.fit_transform(df['Physically_attacked'])
    df['Physical_fighting'] = le.fit_transform(df['Physical_fighting'])
    df['Felt_lonely'] = le.fit_transform(df['Felt_lonely'])
    df['Close_friends'] = le.fit_transform(df['Close_friends'])
    df['Miss_school_no_permission'] = le.fit_transform(df['Miss_school_no_permission'])
    df['Other_students_kind_and_helpful'] = le.fit_transform(df['Other_students_kind_and_helpful'])
    df['Parents_understand_problems'] = le.fit_transform(df['Parents_understand_problems'])
    df['Most_of_the_time_or_always_felt_lonely'] = le.fit_transform(df['Most_of_the_time_or_always_felt_lonely'])
    df['Missed_classes_or_school_without_permission'] = le.fit_transform(df['Missed_classes_or_school_without_permission'])
    df['Were_underweight'] = le.fit_transform(df['Were_underweight'])
    df['Were_overweight'] = le.fit_transform(df['Were_overweight'])
    df['Were_obese'] = le.fit_transform(df['Were_obese'])
    df.drop('record', axis=1, inplace=True)
    df.drop('Felt_lonely', axis=1, inplace=True)

    X = df.drop('Bullied_on_school_property_in_past_12_months', axis=1)
    y = df['Bullied_on_school_property_in_past_12_months']

    # Apply SelectKBest to extract top 10 best features
    best_features = SelectKBest(score_func=chi2, k=10)
    fit = best_features.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # Concatenate two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # Naming the dataframe columns

    # Select the top 9 features
    top_features = featureScores.nlargest(8, 'Score')['Specs'].tolist()
    X_selected = X[top_features]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Create a Logistic Regression classifier
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Streamlit UI
    st.write("Please provide the following details:")

    # Input fields for features
    physically_attacked = st.selectbox("Have you been physically attacked?", ["Yes", "No"])
    cyber_bullied = st.selectbox("Have you been bullied online by other students in the past year?", ["Yes", "No"])
    felt_lonely = st.selectbox("Did you feel lonely at school most of the time?", ["Yes", "No"])
    kind_helpful = st.selectbox("Do you find other students at school to be kind and helpful?", ["Yes", "No"])
    physical_fight = st.selectbox("Have you been in a physical fight at school?", ["Yes", "No"])
    parents_understand = st.selectbox("Do you feel like your parents understand the problems you face?", ["Yes", "No"])
    miss_school = st.selectbox("Have you skipped school without your parents' permission?", ["Yes", "No"])
    sex = st.selectbox("Gender?", ["Male", "Female"])
    col1, col2, col3 = st.columns(3)
    with col1:
        vicitm=st.checkbox("Victim?")
    with col2:
        parents=st.checkbox("Parents?")
    with col3:
        advisor=st.checkbox("Student advisor?")
      

    # Convert input to numerical values
    input_data = {
        'Physically_attacked': 1 if physically_attacked == "Yes" else 0,
        'Cyber_bullied_in_past_12_months': 1 if cyber_bullied == "Yes" else 0,
        'Most_of_the_time_or_always_felt_lonely': 1 if felt_lonely == "Yes" else 0,
        'Other_students_kind_and_helpful': 1 if kind_helpful == "Yes" else 0,
        'Physical_fighting': 1 if physical_fight == "Yes" else 0,
        'Parents_understand_problems': 1 if parents_understand == "Yes" else 0,
        'Miss_school_no_permission': 1 if miss_school == "Yes" else 0,
        'Sex': 0 if sex == "Male" else 1
    }

    input_df = pd.DataFrame([input_data])

    # Make prediction
    if st.button("Predict", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of being bullied

        col1, col2, col3 = st.columns([1, 300, 1])
        with col2:
            if prediction == 1:
                st.write("This student is likely to have been bullied.")
                image_url = r"C:\Users\simo2\Desktop\EVC_Training\Week_2\project2\Sad.png"

                # Display the image using st.image with the URL
                st.image(image_url, width=100)
            else:
                st.write("This student is likely not to have been bullied.")
                image_url = r"C:\Users\simo2\Desktop\EVC_Training\Week_2\project2\Happy.png"

                # Display the image using st.image with the URL
                st.image(image_url, width=100)

            st.write(f"Probability of being bullied: {probability:.2f}")

            # Display most likely reason (this part is simplified and needs further refinement)
            st.write("Most likely reason for bullying:")
            if physically_attacked == "Yes":
                st.write("- Physical attacks")
                
                

            elif cyber_bullied == "Yes":
                st.write("- Cyberbullying")
            elif felt_lonely == "Yes":
                st.write("- Feeling lonely")
            elif kind_helpful == "Yes":
                st.write("- Experiencing difficulties with other students")
            elif physical_fight == "Yes":
                st.write("- Getting into physical fight")
            elif parents_understand == "Yes":
                st.write("- Lack of understanding from parents")
            elif miss_school == "Yes":
                st.write("- Skipping school without permission")
                
            if(vicitm):                         
                youtube_link = "https://youtu.be/iFlrCuSyhvU?si=k7HfVkFnO8LhJ1R"
                st.video(youtube_link)
                
            if(parents):
                youtube_link = "https://youtu.be/mdnTapkLbxs?si=HF_oUbirJMOQ5dH6s"
                st.video(youtube_link)
                
            if(advisor):
                youtube_link = "https://youtu.be/FFD6ik1Edec?si=QlDDR_jMBhIl76rQ"
                st.video(youtube_link)

# Ensure the function is called when this script is run
if __name__ == '__main__':
    page2()
