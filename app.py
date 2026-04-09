import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.preprocess import clean_text
from src.keywords import extract_keywords 
from src.sentiment import analyze_text

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("💬 Customer Sentiment Dashboard")
st.markdown("Analyze customer feedback with AI insights 🚀")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.iloc[:, 0:1]
    df.columns = ['review']

    df = df.dropna(subset=['review'])
    st.info("Processing ")

    df['cleaned']=df['review'].apply(clean_text)
    df['sentiment']=df['cleaned'].apply(lambda x:analyze_text(x)['label'])
    df['score']=df['cleaned'].apply(lambda x:analyze_text(x)['score'])

    df['keywords']=df['cleaned'].apply(lambda x:  extract_keywords(x))

    st.success('Analysis Completed ')

    positive = (df['sentiment']=="Positive").sum()
    negative = (df['sentiment']=="Negative").sum()
    neutral = (df['sentiment']=="Neutral").sum()

    col1 , col2 , col3 , col4 = st.columns(4)

    col1.metric("Total Reviews ",len(df))
    col2.metric("Positive",positive)
    col3.metric("Negative",negative)
    col4.metric("Neutral",neutral)


    st.subheader("Data Preview")
    st.dataframe(df.head())


    st.subheader("Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())


    st.subheader("Sentiment Breakdown")

    counts = df['sentiment'].value_counts()


    fig , ax = plt.subplots()
    ax.pie(counts,labels=counts.index,autopct='%1.1f%%')
    st.pyplot(fig)


    #------------WORD CLOUD -------
    st.subheader("Word Cloud ")

    text = " ".join(df['cleaned'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")

    st.pyplot(fig_wc)

    # ---------------- SUMMARY ---------------- #
    st.subheader("🧠 Summary Insights")

    if positive > negative:
        st.success("Customers are mostly satisfied with the product.")
    elif negative > positive:
        st.error("Customers have more negative feedback. Improvements needed.")
    else:
        st.warning("Customer sentiment is mixed.")
