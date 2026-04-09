import streamlit as st 
from keybert import KeyBERT


@st.cache_resource
def load_model():
    return KeyBERT()

kw_model = load_model()

def extract_keywords(text):
    if not text or len(text.strip())==0:
        return[]
    
    keywords=kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,2),
        stop_words="english",
        top_n=5
    )