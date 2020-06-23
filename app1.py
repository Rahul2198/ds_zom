# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:47:58 2020

@author: hp-pc
"""


# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
import time
df = pd.read_csv('for_model.csv')
st.title('Normalized Rating Predictor ')
k = load_model('lstm_model.h5')
st.subheader('This is a simple application which helps in predicting the ratings for the given user review.')
hotel=st.text_input("Enter the name of the Restaurant", )

if st.button("Next"):
    result= hotel.title()
    st.success(result)
   
review=st.text_input("Your Reviews and Comments", )

#if st.button("Submit"):
   #with st.spinner("Review Processing....") :
       #time.sleep(5)
       #st.success("Review Processed")
       #st.button('Predict your Rating')



def funt(r):
   
     
    
     review2 = re.sub('[^a-zA-Z]', ' ', r)
     review2 = review2.lower()
     review2 = review2.split()
  
     review2 = [lemmatizer.lemmatize(word) for word in review2 if not word in stopwords.words('english')]
     review2 = ' '.join(review2)
     a=[]
     a.append(review2)
     voc_size=10000

     onehot_repr=[one_hot(word,voc_size) for word in a] 
     

       
    
     sent_length=50
     embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
     X_final=np.array(embedded_docs)
     return X_final
 
review1 = funt(review)
if st.button('Submit and Predict your Rating'):
    prediction = k.predict_classes(review1)
    st.balloons()
    st.success(f'Your rating for the given review is {(prediction[0])} ')

