import streamlit as st
import requests
import json
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

st.title("Student E-Welcome Desk")
st.subheader("You are a student? New to France? Ask us all your questions and we will give you an answer")

input_user = st.text_input('Ask your question here', 'Hello')

if st.button('Get Answer'):
    # API call to Code 1
    response = requests.post("http://localhost:5000/predict", json={"input_user": input_user})
    result = response.json()["result"]

    # Display prediction
    st.write(f"The answer is :{result}")
    
    