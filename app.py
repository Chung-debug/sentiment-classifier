import streamlit as st
from transformers import pipeline

st.title("Sentiment Classifier")
# You'll need to unzip the model file in your code or upload the folder unzipped
classifier = pipeline("sentiment-analysis", model="./sentiment_model")

text = st.text_input("Enter a review:")
if text:
    result = classifier(text)
    st.write(f"Sentiment: {result[0]['label']} (Score: {result[0]['score']:.4f})")
