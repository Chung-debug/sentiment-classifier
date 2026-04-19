import streamlit as st
from transformers import pipeline

st.title("Sentiment Classifier")

# Load directly from Hugging Face Hub
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="Chung720/my-sentiment-model")

classifier = load_model()
text = st.text_input("Enter a review:")

if text:
    result = classifier(text)[0]
    st.write(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
