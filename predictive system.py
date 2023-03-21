# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:30:45 2023

@author: sajid
"""

import numpy as np
import pickle


loaded_model = pickle.load(open('C:/Users/sajid/ANISA/Machine learning projects/lung cancer/lungcancer.sav','rb'))


input_data = (1,68,1,2,2,21,1,2,1,2,2,2,2,2,2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] ==0):
    print('The person has  no lung cancer')
else:
    print('The person has lung cancer')