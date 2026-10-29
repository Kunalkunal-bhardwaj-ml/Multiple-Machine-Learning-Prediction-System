

import numpy as np
import pandas as pd
import pickle 
import streamlit as st
from streamlit_option_menu import option_menu

# load trained model
diabetes_model=pickle.load(open('Diabetes_model.sav','rb'))
hear_disease_model=pickle.load(open('heart_disease_model.sav','rb'))
spam_mail_model=pickle.load(open('spam_mail_model.sav','rb'))


# sidewar for nevigate
with st.sidebar:
    
    selected=option_menu('Multiple Model Prediction System',['Diabetes Prediction',
                                                              'Heart Disease Prediction',
                                                              'Spam Mail'],
                                                               default_index=0)
    
 

# prediction page

if(selected=='Diabetes Prediction'):
    
    st.title('Diabetes Prediction Using Ml')
    
    # input data of diabetes prediction model
    col1 ,col2 ,col3 = st.columns(3)
    
    with col1:
        Pregnancies=st.text_input("Number of pregnancies")
        
    with col2:    
        Glucose=st.text_input("Enter Glucose level")
        
    with col3:    
        BloodPressure=st.text_input("Blood-Pressure value")
        
    with col1:     
        SkinThickness=st.text_input("Enter skin_Thickness")
     
    with col2:    
         Insulin=st.text_input("Insuline level")
         
    with col3:     
         BMI=st.text_input("Enter BMI value")
         
    with col1:     
          DiabetesPedigreeFunction=st.text_input("Enter Diabetes_Pedigree_Function")
          
    with col2:      
          Age=st.text_input("Enter Age")
    
    
    
    # code for prediction
    diab_diagnosis=""
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diab_prediction=diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
     
        if (diab_prediction[0]==1):
            diab_diagnosis="The person is diabetic"
        
        else:
            diab_diagnosis="The person is not diabetic"
        
    st.success(diab_diagnosis)    
        
# Heart disease  
    
    
    
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction Using ML')

    # Load saved encoders
    heart_encoders = pickle.load(
        open('D:/deployment model/heart_disease_encoders.sav', 'rb')
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=52)

        sex = st.selectbox('Sex', ['Male', 'Female'])

        chest_pain_type = st.selectbox(
            'Chest Pain Type',
            ['Typical angina', 'Atypical angina',
             'Non-anginal pain', 'Asymptomatic']
        )

        resting_blood_pressure = st.number_input(
            'Resting Blood Pressure', min_value=0, value=125
        )

        cholestoral = st.number_input(
            'Cholesterol', min_value=0, value=212
        )

    with col2:
        fasting_blood_sugar = st.selectbox(
            'Fasting Blood Sugar',
            ['Lower than 120 mg/ml', 'Greater than 120 mg/ml']
        )

        rest_ecg = st.selectbox(
            'Rest ECG',
            ['ST-T wave abnormality', 'Normal',
             'Left ventricular hypertrophy']
        )

        Max_heart_rate = st.number_input(
            'Maximum Heart Rate', min_value=0, value=168
        )

        exercise_induced_angina = st.selectbox(
            'Exercise Induced Angina', ['No', 'Yes']
        )

        oldpeak = st.number_input('Oldpeak', value=1.0)

    with col3:
        slope = st.selectbox(
            'Slope', ['Downsloping', 'Upsloping', 'Flat']
        )

        vessels_colored_by_flourosopy = st.selectbox(
            'Vessels Colored by Fluoroscopy',
            ['Zero', 'One', 'Two', 'Three', 'Four']
        )

        thalassemia = st.selectbox(
            'Thalassemia',
            ['Reversable Defect', 'Fixed Defect', 'Normal']
        )

    if st.button('Heart Disease Test Result'):

        try:
            # Convert categorical inputs into numbers
            sex = heart_encoders['sex'].transform([sex])[0]

            chest_pain_type = heart_encoders[
                'chest_pain_type'
            ].transform([chest_pain_type])[0]

            fasting_blood_sugar = heart_encoders[
                'fasting_blood_sugar'
            ].transform([fasting_blood_sugar])[0]

            rest_ecg = heart_encoders[
                'rest_ecg'
            ].transform([rest_ecg])[0]

            exercise_induced_angina = heart_encoders[
                'exercise_induced_angina'
            ].transform([exercise_induced_angina])[0]

            slope = heart_encoders['slope'].transform([slope])[0]

            vessels_colored_by_flourosopy = heart_encoders[
                'vessels_colored_by_flourosopy'
            ].transform([vessels_colored_by_flourosopy])[0]

            thalassemia = heart_encoders[
                'thalassemia'
            ].transform([thalassemia])[0]

            # Arrange input in training column order
            heart_input = np.array([[
                age,
                sex,
                chest_pain_type,
                resting_blood_pressure,
                cholestoral,
                fasting_blood_sugar,
                rest_ecg,
                Max_heart_rate,
                exercise_induced_angina,
                oldpeak,
                slope,
                vessels_colored_by_flourosopy,
                thalassemia
            ]], dtype=float)

            # Predict
            heart_prediction = hear_disease_model.predict(heart_input)

            if heart_prediction[0] == 1:
                st.warning(
                    'The model predicts a possibility of heart disease.'
                )
            else:
                st.success(
                    'The model does not predict heart disease.'
                )

        except ValueError as e:
            st.error(f'Input encoding error: {e}')

    
    
    
   
if(selected == 'Spam Mail'):
    
    st.title('Spam Mail Using Ml')
    
    # Get input
    Text_Message=st.text_input("Enter Message")
    
    # load Tf-Idf vectorizer 
    vectorizer=pickle.load(open('D:/deployment model/spam_mail_vectorizer.sav','rb'))
    
    # conver input into suitable format
    Message=vectorizer.transform([Text_Message])
    
    # code for prediction
    mail_result=''
    
    #create a button for prediction
    if st.button('Spam Mail Result'):
        mail_prediction=spam_mail_model.predict([Message])
        
        if(mail_prediction[0]==1):
            mail_result="Spam Mail"
            
        else:
            mail_result="Not spam Mail"
            
    st.success(mail_result)   
    
        
    
    
    