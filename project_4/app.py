import streamlit as st
import numpy as np
from pickle import load
import sklearn

import os
# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath('__file__'))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, 'resources','models','standard_scaler1.pkl')
# absolute path of directory_of_interest
# dir_of_interest = os.path.join(PARENT_DIR, "resources")

# a= os.path.join(dir_of_interest,'models','standard_scaler1.pkl')
b = os.path.join(FILE_DIR, 'resources','models','ohe1.pkl')
c = os.path.join(FILE_DIR, 'resources','models','gbdt_regressor1.pkl')
# b= os.path.join(dir_of_interest,'models','ohe1.pkl')
# c= os.path.join(dir_of_interest,'models','gbdt_regressor1.pkl')

scaler = load(open(PARENT_DIR,'rb'))
ohe = load(open(b,'rb'))
gbdt_model = load(open(c,'rb'))

st.set_page_config(page_title="Insurance Cost Prediction App")

age = st.text_input('Age', placeholder='Enter age')
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.text_input('BMI', placeholder='Enter BMI')
children = st.text_input('Children', placeholder='Enter number of children')
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

btn_click = st.button('Predict')


if btn_click:
    # Validate input values
    if not (age.isnumeric() and bmi.isnumeric() and children.isnumeric()):
        st.error('Age, BMI and Children must be numbers')
    elif sex not in ['male', 'female'] or smoker not in ['yes', 'no'] or region not in ['southwest', 'southeast', 'northwest', 'northeast']:
        st.error('Invalid input')
    else:
        # Convert input to a numpy array
        num = np.array([float(age), float(bmi), float(children)]).reshape(1,-1)
        
        # Scale numerical features
        num_rescaled = scaler.transform(num)
        
        # Encode categorical features
        cat = np.array([[sex, smoker, region]])
        cat_transformed = ohe.transform(cat)
        
        
        # Combine the numerical and categorical features
        query_transformed = np.concatenate([num_rescaled, cat_transformed], axis=1)
        
        # Make prediction
        pred = gbdt_model.predict(query_transformed)
        
        # Show prediction result
        st.success(f'The predicted insurance cost is {pred[0]:.2f} rupees.')
