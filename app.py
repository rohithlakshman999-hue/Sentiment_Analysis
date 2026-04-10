import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.preprocess import clean_text
from src.sentiment import analyze_text
from src.keywords import extract_keywords

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("💬 Customer Sentiment Dashboard")
st.markdown("Analyze customer feedback with AI insights 🚀")

# ---------------- CACHE DATA ---------------- #
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = load_data(uploaded_file)

    if st.button("Start Analysis 🚀"):

        with st.spinner("Processing data... ⏳"):

            # Limit data (IMPORTANT)
            df = df.head(7000)

            # Select column
            df = df.iloc[:, 0:1]
            df.columns = ['review']
            df = df.dropna(subset=['review'])

            # Cleaning
            df['cleaned'] = df['review'].apply(clean_text)

            # Sentiment
            df['sentiment'] = df['cleaned'].apply(lambda x: analyze_text(x)['label'])
            df['score'] = df['cleaned'].apply(lambda x: analyze_text(x)['score'])

            # 🔥 KEY CHANGE: keywords only once
            all_text = " ".join(df['cleaned'])
            keywords = extract_keywords(all_text)

        st.success("Analysis Completed ✅")

        # ---------------- METRICS ---------------- #
        positive = (df['sentiment'] == "Positive").sum()
        negative = (df['sentiment'] == "Negative").sum()
        neutral = (df['sentiment'] == "Neutral").sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", len(df))
        col2.metric("Positive", positive)
        col3.metric("Negative", negative)
        col4.metric("Neutral", neutral)

        # ---------------- FILTER ---------------- #
        st.subheader("🔍 Filter Reviews")

        selected = st.selectbox("Select Sentiment", ["All", "Positive", "Negative", "Neutral"])

        if selected != "All":
            filtered_df = df[df['sentiment'] == selected]
        else:
            filtered_df = df

        # ---------------- TABLE ---------------- #
        st.subheader("📊 Data Preview")
        st.dataframe(filtered_df.head())

        # ---------------- CHARTS ---------------- #
        colA, colB = st.columns(2)

        with colA:
            st.subheader("📈 Bar Chart")
            st.bar_chart(df['sentiment'].value_counts())

        with colB:
            st.subheader("🥧 Pie Chart")
            counts = df['sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(fig)

        # ---------------- WORD CLOUD ---------------- #
        st.subheader("☁️ Word Cloud")

        text = " ".join(df['cleaned'])

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")

        st.pyplot(fig_wc)

        # ---------------- KEYWORDS ---------------- #
        st.subheader("🔑 Top Keywords")
        st.write(keywords)

        # ---------------- SUMMARY ---------------- #
        st.subheader("🧠 Summary Insights")

        total = len(df)

        if positive > negative:
            st.success(f"{(positive/total)*100:.1f}% customers are satisfied.")
        elif negative > positive:
            st.error(f"{(negative/total)*100:.1f}% customers reported issues.")
        else:
            st.warning("Customer sentiment is balanced.")

        # ---------------- DOWNLOAD ---------------- #
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Results",
            data=csv,
            file_name="sentiment_analysis.csv",
            mime="text/csv"
        )