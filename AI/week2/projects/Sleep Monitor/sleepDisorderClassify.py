import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data
sleep_df = pd.read_csv('D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\Sleep_health_and_lifestyle_dataset.csv')
student_df = pd.read_csv('D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\Student_performance_data _.csv')

sleep_df.drop(['Person ID', 'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Stress Level'], axis=1, inplace=True)
sleep_df.fillna('None', inplace=True)

class_names = student_df['GradeClass'].unique()
df_sampled = []

for class_name in class_names:
    sampled_rows = student_df[student_df['GradeClass'] == class_name].head(75)
    df_sampled.append(sampled_rows)

df_sampled = pd.concat(df_sampled, ignore_index=True)

# Ensure the lengths match
if len(sleep_df) > len(df_sampled):
    sleep_df = sleep_df.iloc[:len(df_sampled)]
elif len(sleep_df) < len(df_sampled):
    df_sampled = df_sampled.iloc[:len(sleep_df)]

sleep_df[['Absences', 'ParentalSupport']] = df_sampled.loc[:len(sleep_df)-1, ['Absences', 'ParentalSupport']]
sleep_df['StudyTimeWeekly'] = df_sampled.loc[:len(sleep_df)-1, 'StudyTimeWeekly']
sleep_df['Grade Class'] = df_sampled['GradeClass'].values
sleep_df.drop('Occupation', axis=1, inplace=True)

# Encoding
LE_gender = LabelEncoder()
LE_bmi = LabelEncoder()
LE_sleep_disorder = LabelEncoder()
LE_grade_class = LabelEncoder()  # Add this line to initialize LabelEncoder for Grade Class

# Fit the encoders on the full range of possible values
LE_gender.fit(['Female', 'Male'])
LE_bmi.fit(sleep_df['BMI Category'])
LE_sleep_disorder.fit(sleep_df['Sleep Disorder'])
LE_grade_class.fit(sleep_df['Grade Class'])  # Fit with all possible values in sleep_df['Grade Class']

# Transform the data
sleep_df['Gender'] = LE_gender.transform(sleep_df['Gender'].map({'Female': 'Female', 'Male': 'Male'}))
sleep_df['BMI Category'] = LE_bmi.transform(sleep_df['BMI Category'])
sleep_df['Sleep Disorder'] = LE_sleep_disorder.transform(sleep_df['Sleep Disorder'])
sleep_df['Grade Class'] = LE_grade_class.transform(sleep_df['Grade Class'])  # Encode Grade Class

X = sleep_df.drop('Sleep Disorder', axis=1)
y = sleep_df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')
log_reg.fit(X_train, y_train)

# Streamlit app
st.title('Sleep and Student Performance Classification')

# User input
gender = st.selectbox('Gender', options=LE_gender.classes_)
age = st.number_input('Age', min_value=int(sleep_df['Age'].min()), max_value=int(sleep_df['Age'].max()), step=1)
sleep_duration = st.number_input('Sleep Duration', min_value=float(sleep_df['Sleep Duration'].min()), max_value=float(sleep_df['Sleep Duration'].max()), step=0.1)
quality_of_sleep = st.number_input('Quality of Sleep', min_value=float(sleep_df['Quality of Sleep'].min()), max_value=float(sleep_df['Quality of Sleep'].max()), step=0.1)
physical_activity = st.selectbox('Physical Activity Level', options=sleep_df['Physical Activity Level'].unique())
bmi_category = st.selectbox('BMI Category', options=LE_bmi.classes_)

absences = st.number_input('Absences', min_value=int(sleep_df['Absences'].min()), max_value=int(sleep_df['Absences'].max()), step=1)
parental_support = st.selectbox('Parental Support', options=sleep_df['ParentalSupport'].unique())
study_time_weekly = st.number_input('Study Time Weekly', value=9.202053, step=0.1)
grade_class = st.selectbox('Grade Class', options=LE_grade_class.classes_)

# Encode user input
user_data = {
    'Gender': LE_gender.transform([gender])[0],
    'Age': age,
    'Sleep Duration': sleep_duration,
    'Quality of Sleep': quality_of_sleep,
    'Physical Activity Level': physical_activity,
    'BMI Category': LE_bmi.transform([bmi_category])[0],
    
    'Absences': absences,
    'ParentalSupport': parental_support,
    'StudyTimeWeekly': study_time_weekly,
    'Grade Class': LE_grade_class.transform([grade_class])[0]
}

# Convert user input to DataFrame
user_df = pd.DataFrame(user_data, index=[0])

# Scale user input
user_scaled = scaler.transform(user_df)

# Predict classification
if st.button('Classify'):
    prediction = log_reg.predict(user_scaled)
    if(prediction[0]==0):
        st.write(f'The predicted Sleep Disorder class is: Healthy')
    elif(prediction[0]==1):
        st.write(f'The predicted Sleep Disorder class is: Sleep Apnea')
    else:
        st.write(f'The predicted Sleep Disorder class is: Insomnia')
