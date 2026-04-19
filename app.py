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
    prediction = classifier(text)[0] # Get the first result
    
    # Create the mapping
    # Note: If your model outputs "LABEL_1", use that as the key instead
    label_map = {"1": "Positive", "0": "Negative", "LABEL_1": "Positive", "LABEL_0": "Negative"}
    
    # Get the clean label
    raw_label = prediction['label']
    clean_label = label_map.get(raw_label, raw_label)
    
    score = prediction['score']
    
    st.write(f"**Result:** {clean_label} (Confidence: {score:.2f})")
