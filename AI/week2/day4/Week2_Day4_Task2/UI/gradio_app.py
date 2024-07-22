import gradio as gr
import joblib

# Load the trained model
# u have to replace this with ur path
model = joblib.load(r'C:\Users\hebah\Documents\EVCProjects\W2D4T2\model\insurance_charges_model.pkl')

# Define the prediction function
def predict_charges(age, sex, bmi, children, smoker, region):
    sex = 0 if sex == 'male' else 1
    smoker = 1 if smoker == 'yes' else 0
    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0

    features = [age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest]
    return model.predict([features])[0]

# Gradio interface
inputs = [
    gr.Number(label='Age'),
    gr.Radio(['male', 'female'], label='Sex'),
    gr.Number(label='BMI'),
    gr.Number(label='Number of children'),
    gr.Radio(['yes', 'no'], label='Smoker'),
    gr.Radio(['northwest', 'northeast', 'southeast', 'southwest'], label='Region')
]

outputs = gr.Textbox(label='Estimated Insurance Charges')

gr.Interface(fn=predict_charges, inputs=inputs, outputs=outputs, title='Insurance Charges Prediction').launch()
