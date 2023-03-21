# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:37:27 2023

@author: sajid
"""

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('C:/Users/sajid/ANISA/Machine learning projects/lung cancer/lungcancer.sav','rb'))


    
    
def main():
    st.title('lung cancer prediction web app')
    
       
    GENDER = st.text_input("gender of person(please mention gender type: male,female)")
    AGE = st.text_input('age of person')
    SMOKING = st.text_input('smoking')
    YELLOW_FINGERS= st.text_input('yellow_fingers')
    ANXIETY = st.text_input('anxiety')
    PEER_PRESSURE = st.text_input('peer_pressure')
    CHRONIC_DISEASE = st.text_input('chronic_disease')
    FATIGUE = st.text_input('fatigue')
    ALLERGY = st.text_input('allergy')
    WHEEZING = st.text_input('wheezing')
    ALCOHOL_CONSUMING = st.text_input('alcohol_consuming')
    COUGHING = st.text_input('coughing')
    SHORTNESS_OF_BREATH = st.text_input('shortness_of_breath')
    SWALLOWING_DIFFICULTY = st.text_input('swallowing_difficulty')
    CHEST_PAIN = st.text_input('chest_pain')
    
    
    lungcancer = ''
    
    
    if st.button("predict"):
        lungcancer = loaded_model.predict([[GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]])
        
        
        
        
    st.success(lungcancer)
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
    
        