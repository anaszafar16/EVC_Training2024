import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
 # Load the data
train_data = pd.read_csv("C:\\Users\\abdul\\Downloads\\train.csv")
test_data = pd.read_csv("C:\\Users\\abdul\\Downloads\\test.csv")

# Preprocess the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Split the data into features and labels
y_train = train_data['label']
y_test = test_data['label']

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create the Streamlit app
st.title('Emotion Classification')

# Text input
text = st.text_area('Enter your text:', '')

# Preprocess the input text
input_text = vectorizer.transform([text])

# Predict the emotion
if st.button('Classify'):
    prediction = model.predict(input_text)[0]
    emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    st.write(f'The emotion detected is: {emotion_labels[prediction]}')

# Evaluation metrics
st.subheader('Model Evaluation')
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f'Accuracy: {accuracy:.2f}')