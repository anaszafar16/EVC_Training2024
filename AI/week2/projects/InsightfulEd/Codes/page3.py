import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st

def page3():
    # Read the data into a pandas DataFrame
    data = pd.read_csv(r"C:\Users\simo2\Desktop\EVC_Training\Week_2\project2\adult.csv")

    # Replace (?) with NaNs
    data.replace('?', np.nan, inplace=True)

    # Select relevant features
    selected_features = ['age', 'marital-status', 'occupation', 'educational-num', 'hours-per-week']

    # Split hours-per-week into fulltime and parttime
    data['hours-per-week'] = pd.cut(data['hours-per-week'], bins=[0, 39, data['hours-per-week'].max()], labels=['parttime', 'fulltime'])

    # Create feature matrix X and target vector y
    X = data[selected_features]
    y = data['income']  # Assuming 'income' is the target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    numeric_features = ['age', 'educational-num']
    categorical_features = ['marital-status', 'occupation', 'hours-per-week']

    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the pipeline for Random Forest Classifier
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    pipeline_rf.fit(X_train, y_train)

    # Save the model
    joblib.dump(pipeline_rf, 'model.joblib')

    # Evaluation (Optional, remove if not needed)
    # rf_predictions = pipeline_rf.predict(X_test)

    # Load the trained model
    model = joblib.load('model.joblib')

    # Streamlit app
    st.title('Income Prediction')

    # Feature explanations
    st.write('Answer the questions to predict your salary')

    # Age slider
    age = st.slider('Select age:', min_value=17, max_value=90, value=30)

    # Marital status dropdown
    marital_status_options = [
        'Never-married',
        'Married-civ-spouse',
        'Divorced',
        'Separated',
        'Widowed',
        'Married-spouse-absent',
        'Married-AF-spouse'
    ]
    marital_status = st.selectbox('Select marital status:', options=marital_status_options)

    # Occupation dropdown
    occupation_options = [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ]
    occupation = st.selectbox('Select occupation:', options=occupation_options)

    # Educational num slider
    educational_num = st.slider('Select educational number:', min_value=1, max_value=16, value=10)

    # Hours per week radio buttons
    hours_per_week = st.radio('Select hours per week:', ['parttime', 'fulltime'])

    # Create input data based on user selections
    input_data = {
        'age': age,
        'marital-status': marital_status,
        'occupation': occupation,
        'educational-num': educational_num,
        'hours-per-week': hours_per_week
    }

    # Predict function
    def predict_income(data):
        # Prepare input data for prediction
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return prediction[0]

    # Prediction button
    if st.button('Predict Income Class'):
        prediction = predict_income(input_data)
        income_class = '>=50K' if prediction == 1 else '<50K'
        st.header(f'The predicted income : {income_class}')

if __name__ == '__main__':
    page3()
