import streamlit as st
from transformers import pipeline

st.title("Sentiment Classifier 2")

st.write("""
Traditional NLP models, like the one shown in the screenshot below, often rely on 'Bag of Words'. 
They don't understand the relationship between words; they only see that 'bad' is usually a negative word. 
""")

st.image("rev1.png", 
         caption="Old Model: Fails to understand the context of slang like 'badass'.",
         use_container_width=True)

st.markdown("""
Unlike traditional classifiers that simply look for "good" or "bad" words, this model uses **Transfer Learning** with a pre-trained **Transformer (BERT)**. 

**The Difference:**
*   **Traditional NLP:** Often fails on slang or sarcasm. For example, it sees "badass" and flags it as **Negative** because of the word "bad".
*   **This Model:** Understands nuance. It recognizes that "badass" is a compliment, resulting in a **Positive** classification.
""")


# Load directly from Hugging Face Hub
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="Chung720/my-sentiment-model")

classifier = load_model()
text = st.text_input("Enter a review:")

if text:
    prediction = classifier(text)[0] # Get the first result
    
    # Create the mapping
    label_map = {"1": "Positive", "0": "Negative", "LABEL_1": "Positive", "LABEL_0": "Negative"}
    
    # Get the clean label
    raw_label = prediction['label']
    clean_label = label_map.get(raw_label, raw_label)
    
    score = prediction['score']
    
    st.write(f"**Result:** {clean_label} (Confidence: {score:.2f})")
