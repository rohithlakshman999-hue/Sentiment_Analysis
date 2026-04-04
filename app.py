import streamlit as st
import pandas as pd
from src.preprocess import clean_text
from src.sentiment import analyze_sentiment
from src.keywords import extract_keywords

st.title("💬 Sentiment Analysis App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("File uploaded successfully ✅")

    df = pd.read_csv(uploaded_file)

    # Show columns
    st.write("Columns:", df.columns)

    # Always take first column
    df = df.iloc[:, 0:1]
    df.columns = ['review']

    st.write("Processing... ⏳")

    df['cleaned'] = df['review'].apply(clean_text)
    df['sentiment'] = df['cleaned'].apply(analyze_sentiment)
    df['keywords'] = df['cleaned'].apply(lambda x: extract_keywords(x))

    st.success("Analysis Completed ✅")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Sentiment Count")
    st.bar_chart(df['sentiment'].value_counts())