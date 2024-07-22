import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose an option:",
        ["Home", "Class Grade Classification", "Predict Sleep Duration", "Sleep Disorder Classification"]
    )

    if app_mode == "Home":
        show_home()
    elif app_mode == "Class Grade Classification":
        show_grade_classification()
    elif app_mode == "Predict Sleep Duration":
        show_sleep_duration_prediction()
    elif app_mode == "Sleep Disorder Classification":
        show_sleep_disorder_classification()

def show_home():
    st.title("Welcome to Dream App")
    st.image("D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\pic.png", use_column_width=True, caption="Dream App Home")

def show_grade_classification():
    st.title('Sleep and Student Performance Classification')
    st.subheader('Class Grade Classification')

    sleep_df = pd.read_csv('D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\Sleep_health_and_lifestyle_dataset.csv')
    student_df = pd.read_csv('D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\Student_performance_data _.csv')

    sleep_df.drop(['Person ID', 'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Stress Level'], axis=1, inplace=True)
    sleep_df.fillna('None', inplace=True)

    class_names = student_df['GradeClass'].unique()
    df_sampled = [student_df[student_df['GradeClass'] == class_name].head(75) for class_name in class_names]
    df_sampled = pd.concat(df_sampled, ignore_index=True)

    if len(sleep_df) > len(df_sampled):
        sleep_df = sleep_df.iloc[:len(df_sampled)]
    elif len(sleep_df) < len(df_sampled):
        df_sampled = df_sampled.iloc[:len(sleep_df)]

    sleep_df[['Absences', 'ParentalSupport']] = student_df.loc[:len(sleep_df)-1, ['Absences', 'ParentalSupport']]
    sleep_df['StudyTimeWeekly'] = student_df.loc[:len(sleep_df)-1, 'StudyTimeWeekly']
    sleep_df['Grade Class'] = df_sampled['GradeClass'].values
    sleep_df.drop('Occupation', axis=1, inplace=True)

    LE_gender = LabelEncoder()
    LE_bmi = LabelEncoder()
    LE_sleep_disorder = LabelEncoder()

    LE_gender.fit(['Female', 'Male'])
    LE_bmi.fit(sleep_df['BMI Category'])
    LE_sleep_disorder.fit(sleep_df['Sleep Disorder'])

    sleep_df['Gender'] = LE_gender.transform(sleep_df['Gender'].map({'Female': 'Female', 'Male': 'Male'}))
    sleep_df['BMI Category'] = LE_bmi.transform(sleep_df['BMI Category'])
    sleep_df['Sleep Disorder'] = LE_sleep_disorder.transform(sleep_df['Sleep Disorder'])

    X = sleep_df.drop('Grade Class', axis=1)
    y = sleep_df['Grade Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    gender = st.selectbox('Gender', options=LE_gender.classes_)
    age = st.number_input('Age', min_value=int(sleep_df['Age'].min()), max_value=int(sleep_df['Age'].max()), step=1)
    sleep_duration = st.number_input('Sleep Duration', min_value=float(sleep_df['Sleep Duration'].min()), max_value=float(sleep_df['Sleep Duration'].max()), step=0.1)
    physical_activity = st.selectbox('Physical Activity Level', options=sleep_df['Physical Activity Level'].unique())
    bmi_category = st.selectbox('BMI Category', options=LE_bmi.classes_)
    sleep_disorder = st.selectbox('Sleep Disorder', options=LE_sleep_disorder.classes_)
    absences = st.number_input('Absences', min_value=int(sleep_df['Absences'].min()), max_value=int(sleep_df['Absences'].max()), step=1)
    study_time_weekly = st.number_input('Study Time Weekly', value=9.202053, step=0.1)

    fixed_quality_of_sleep = 7.5
    fixed_parental_support = 3

    user_data = {
        'Gender': LE_gender.transform([gender])[0],
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': fixed_quality_of_sleep,  # Fixed value
        'Physical Activity Level': physical_activity,
        'BMI Category': LE_bmi.transform([bmi_category])[0],
        'Sleep Disorder': LE_sleep_disorder.transform([sleep_disorder])[0],
        'Absences': absences,
        'ParentalSupport': fixed_parental_support,  # Fixed value
        'StudyTimeWeekly': study_time_weekly
    }

    user_df = pd.DataFrame(user_data, index=[0])
    user_scaled = scaler.transform(user_df)

    # if st.button('Classify'):
    #     prediction = log_reg.predict(user_scaled)
    #     st.success(f'The predicted Grade Class is: {prediction[0]}')
    grade_class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

    # Predict classification
    if st.button('Classify'):
        prediction = log_reg.predict(user_scaled)
        
        if(float(prediction[0])==0.0):
            st.write(f'The predicted Grade is: A')
        elif(float(prediction[0])==1.0):
            st.write(f'The predicted Grade is: B')
        elif(float(prediction[0])==2.0):
            st.write(f'The predicted Grade is: C')
        elif(float(prediction[0])==3.0):
            st.write(f'The predicted Grade is: D')
        elif(float(prediction[0])==4.0):
            st.write(f'The predicted Grade is: F')
        else:
            pass

def show_sleep_duration_prediction():
    import joblib

    grade_class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}

    @st.cache_data
    def load_model():
        model = joblib.load("D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\reg_model.joblib")
        return model

    def predict_sleep_duration(quality_of_sleep, physical_activity, bmi_category, sleep_condition, age, gender, study_time, absences, parental_support, grade_class, model):
        gender_encoded = 0 if gender == "Male" else 1
        grade_class_encoded = list(grade_class_mapping.keys())[list(grade_class_mapping.values()).index(grade_class)]
        sleep_condition_encoded = 0 if sleep_condition == "Sleep Disorder healthy" else (2 if sleep_condition == "Sleep Apnea" else 1)
        bmi_category_encoded = 3 if bmi_category == "Overweight" else (2 if bmi_category == "Obese" else (1 if bmi_category == "Normal Weight" else 0))

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

        prediction = model.predict(data)[0]
        return prediction

    st.title("Sleep Duration Prediction")
    st.write("Enter the following information to predict sleep duration:")

    model = load_model()

    quality_of_sleep = st.slider("Quality of Sleep", 1, 10)
    physical_activity = st.slider("Physical Activity Level (mins)", 30, 90)
    bmi_category = st.selectbox("BMI Category", ["Overweight", "Normal", "Obese", "Normal Weight"])
    sleep_condition = st.selectbox("Sleep Disorder", ["Sleep Disorder healthy", "Sleep Apnea", "Insomnia"])
    age = st.number_input("Age", min_value=1, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    study_time = st.number_input("Study Time (Weekly)", min_value=0, max_value=20, value=10)
    absences = st.number_input("Absences", min_value=0, max_value=30, value=0)
    grade_class = st.selectbox("Grade Class", ["A", "B", "C", "D", "F"])

    fixed_parental_support = 3  

    if st.button("Predict Sleep Duration"):
        sleep_duration = predict_sleep_duration(quality_of_sleep, physical_activity, bmi_category, sleep_condition, age, gender, study_time, absences, fixed_parental_support, grade_class, model)
        st.success(f"Predicted Sleep Duration: {round(sleep_duration, 2)} hours")

def show_sleep_disorder_classification():
    st.title('Sleep Disorder Classification')
    st.subheader('Predict Sleep Disorder')

    sleep_df = pd.read_csv('D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\Sleep_health_and_lifestyle_dataset.csv')
    student_df = pd.read_csv('D:\\Azzam\\Personal_Projects\\EVC\\Python\\AI_Track\\Codes\\streamlit_classification\\Student_performance_data _.csv')

    sleep_df.drop(['Person ID', 'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Stress Level'], axis=1, inplace=True)
    sleep_df.fillna('None', inplace=True)

    class_names = student_df['GradeClass'].unique()
    df_sampled = [student_df[student_df['GradeClass'] == class_name].head(75) for class_name in class_names]
    df_sampled = pd.concat(df_sampled, ignore_index=True)

    if len(sleep_df) > len(df_sampled):
        sleep_df = sleep_df.iloc[:len(df_sampled)]
    elif len(sleep_df) < len(df_sampled):
        df_sampled = df_sampled.iloc[:len(sleep_df)]

    sleep_df[['Absences', 'ParentalSupport']] = df_sampled.loc[:len(sleep_df)-1, ['Absences', 'ParentalSupport']]
    sleep_df['StudyTimeWeekly'] = df_sampled.loc[:len(sleep_df)-1, 'StudyTimeWeekly']
    sleep_df['Grade Class'] = df_sampled['GradeClass'].values
    sleep_df.drop('Occupation', axis=1, inplace=True)

    LE_gender = LabelEncoder()
    LE_bmi = LabelEncoder()
    LE_sleep_disorder = LabelEncoder()
    LE_grade_class = LabelEncoder()

    LE_gender.fit(['Female', 'Male'])
    LE_bmi.fit(sleep_df['BMI Category'])
    LE_sleep_disorder.fit(sleep_df['Sleep Disorder'])
    LE_grade_class.fit(sleep_df['Grade Class'])

    sleep_df['Gender'] = LE_gender.transform(sleep_df['Gender'].map({'Female': 'Female', 'Male': 'Male'}))
    sleep_df['BMI Category'] = LE_bmi.transform(sleep_df['BMI Category'])
    sleep_df['Sleep Disorder'] = LE_sleep_disorder.transform(sleep_df['Sleep Disorder'])
    sleep_df['Grade Class'] = LE_grade_class.transform(sleep_df['Grade Class'])

    X = sleep_df.drop('Sleep Disorder', axis=1)
    y = sleep_df['Sleep Disorder']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')
    log_reg.fit(X_train, y_train)

    gender = st.selectbox('Gender', options=LE_gender.classes_)
    age = st.number_input('Age', min_value=int(sleep_df['Age'].min()), max_value=int(sleep_df['Age'].max()), step=1)
    sleep_duration = st.number_input('Sleep Duration', min_value=float(sleep_df['Sleep Duration'].min()), max_value=float(sleep_df['Sleep Duration'].max()), step=0.1)
    quality_of_sleep = st.number_input('Quality of Sleep', min_value=float(sleep_df['Quality of Sleep'].min()), max_value=float(sleep_df['Quality of Sleep'].max()), step=0.1)
    physical_activity = st.selectbox('Physical Activity Level', options=sleep_df['Physical Activity Level'].unique())
    bmi_category = st.selectbox('BMI Category', options=LE_bmi.classes_)
    absences = st.number_input('Absences', min_value=int(sleep_df['Absences'].min()), max_value=int(sleep_df['Absences'].max()), step=1)
    study_time_weekly = st.number_input('Study Time Weekly', value=9.202053, step=0.1)
    grade_class = st.selectbox('Grade Class', options=LE_grade_class.classes_)

    fixed_parental_support = 3

    user_data = {
        'Gender': LE_gender.transform([gender])[0],
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'BMI Category': LE_bmi.transform([bmi_category])[0],
        'Absences': absences,
        'ParentalSupport': fixed_parental_support,  # Fixed value
        'StudyTimeWeekly': study_time_weekly,
        'Grade Class': LE_grade_class.transform([grade_class])[0]
    }

    user_df = pd.DataFrame(user_data, index=[0])
    user_scaled = scaler.transform(user_df)

    if st.button('Classify'):
        prediction = log_reg.predict(user_scaled)
        if prediction[0] == 0:
            st.write(f'The predicted Sleep Disorder is: None')
        elif prediction[0] == 1:
            st.write(f'The predicted Sleep Disorder is: Sleep Apnea')
        else:
            st.write(f'The predicted Sleep Disorder is: Insomnia')

if __name__ == "__main__":
    main()
