import streamlit as st
from keybert import KeyBERT

# Load model only once (IMPORTANT 🔥)
@st.cache_resource
def load_model():
    return KeyBERT()

kw_model = load_model()

def extract_keywords(text):
    if not text:
        return []

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=5
    )

    return [kw[0] for kw in keywords]