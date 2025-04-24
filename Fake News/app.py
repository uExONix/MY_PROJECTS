# streamlit_app.py

import streamlit as st
import pandas as pd
import string
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Clean text function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Fake News" if n == 0 else "Real News"

def load_models():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
    return vectorizer, model

# Streamlit App
st.title("📰 Fake News Detector")
st.write("Paste any news article below to see if it's fake or real.")

input_news = st.text_area("Enter News Text")

if st.button("Predict"):
    vectorizer, model = load_models()
    cleaned_text = wordopt(input_news)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    label = output_label(prediction[0])
    st.subheader("Prediction:")
    st.success(label)
