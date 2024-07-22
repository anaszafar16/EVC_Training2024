import streamlit as st
import joblib

# Load the trained model
# u have to replace this with ur path
model = joblib.load(r'C:\Users\hebah\Documents\EVCProjects\W2D4T2\model\insurance_charges_model.pkl')

# Define the prediction function
def predict_charges(features):
    return model.predict([features])[0]

# Streamlit app layout
st.title('Insurance Charges Prediction')

age = st.number_input('Age', min_value=0)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input('BMI')
children = st.number_input('Number of children', min_value=0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['northwest', 'northeast', 'southeast', 'southwest'])

# Convert inputs to the appropriate format
sex = 0 if sex == 'male' else 1
smoker = 1 if smoker == 'yes' else 0
region_northwest = 1 if region == 'northwest' else 0
region_southeast = 1 if region == 'southeast' else 0
region_southwest = 1 if region == 'southwest' else 0

features = [age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest]

if st.button('Predict'):
    prediction = predict_charges(features)
    st.write(f'Estimated Insurance Charges: ${prediction:.2f}')
