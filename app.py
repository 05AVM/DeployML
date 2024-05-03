#trying to deploy an ml model using stereamlit
# for fake news detection 
import streamlit as st
import pickle
#import sklearn
import pandas as pd
import numpy as np
from PIL import Image

#from sklearn.feature_extraction.text import TfidfVectorizer 

#loading the model

model=pickle.load(open('./model.sav','rb'))

#from joblib import load

#model = load('./model.joblib')
vectorizer=pickle.load(open('./vectorizer.pkl','rb'))

#creating the app
st.title('FAKE NEWS DETECTION')
#st.sidebar.header('USER INPUT')
image=Image.open('fake news.jpg')
st.image(image,use_column_width=True)

#taking user input
# Add a text input field for user input
user_input = st.text_input("Enter the news text to verify:")
if user_input:

    user_input_transformed=vectorizer.transform([user_input])
# Add a button for users to trigger the prediction
if st.button("Verify"):
    # Check if the user has entered any text
    if user_input:
        # Make prediction using the loaded model
        prediction = model.predict(user_input_transformed)
        
        # Display the prediction result
        st.sidebar.write("## Prediction:")

        if prediction == 1:
            st.error("Fake News Detected!")
        else:
            st.success("Real News Detected!")
    else:
        st.warning("Please enter some news text to verify.")



