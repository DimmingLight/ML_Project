import streamlit as st
import numpy as np
import pandas as pd
import joblib

classifier = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def home():
    st.title('Heart Disease Classification')

    st.write('Please enter the following information:')
    age = st.number_input('Age (in years)', min_value=0, max_value=120, help='Enter your age in years.')
    sex = st.selectbox('Sex', options={0: 'Female', 1: 'Male'}, format_func=lambda x: {0: 'Female', 1: 'Male'}[x], help='Select your gender.')
    cp = st.selectbox('Chest Pain Type', options={0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}, format_func=lambda x: {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}[x], help='Select the type of chest pain experienced.')
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=200, help='Enter your resting blood pressure in mm Hg.')
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600, help='Enter your serum cholesterol level in mg/dl.')
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options={0: 'False', 1: 'True'}, format_func=lambda x: {0: 'False', 1: 'True'}[x], help='Select whether your fasting blood sugar is greater than 120 mg/dl.')
    restecg = st.selectbox('Resting Electrocardiographic Results', options={0: 'Normal', 1: 'Having ST-T wave abnormality', 2: 'Showing probable or definite left ventricular hypertrophy'}, format_func=lambda x: {0: 'Normal', 1: 'Having ST-T wave abnormality', 2: 'Showing probable or definite left ventricular hypertrophy'}[x], help='Select the result of your resting electrocardiogram.')
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=200, help='Enter your maximum heart rate achieved.')
    exang = st.selectbox('Exercise Induced Angina', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x], help='Select whether you experience exercise-induced angina.')
    oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=6.0, help='Enter the ST depression induced by exercise.')
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options={0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}, format_func=lambda x: {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}[x], help='Select the slope of the peak exercise ST segment.')
    satur = st.number_input('Oxygen Saturation', min_value=0.0, max_value=100.0, help='Enter the oxygen saturation results')

    if st.button('Predict'):
        # sex = 1 if sex == 'Male' else 0
        # fbs = 1 if fbs == 'True' else 0
        # exang = 1 if exang == 'Yes' else 0

        user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, satur]])

        user_input_scaled = scaler.transform(user_input)

        prediction = classifier.predict(user_input_scaled)
        if prediction[0] == 0:
            st.markdown('<h4 style="color: white;">The model predicts that the patient does <strong>not</strong> have heart disease.</h4>', unsafe_allow_html=True)
        else:
            st.markdown('<h4 style="color: white;">The model predicts that the patient <strong>has</strong> heart disease.</h4>', unsafe_allow_html=True)
        # if prediction[0] == 0:
        #     st.write('The model predicts that the patient does **not** have heart disease.')
        # else:
        #     st.write('The model predicts that the patient **has** heart disease.')

# def main():
    # pages = {
    #     "Home": home,
    # }

    # st.sidebar.title("Navigation")
    # selection = st.sidebar.radio("Go to", list(pages.keys()))

    # page = pages[selection]
    # page()

if __name__ == "__main__":
    home()