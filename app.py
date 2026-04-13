import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.preprocess import clean_text
from src.sentiment import analyze_text
from src.keywords import extract_keywords
from src.ml_model import predict_sentiment

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("💬 Customer Sentiment Dashboard")
st.markdown("Analyze customer feedback with AI insights 🚀")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = load_data(uploaded_file)

    if st.button("Start Analysis 🚀"):

        with st.spinner("Processing... ⏳"):

            df = df.head(10000)
            df = df.iloc[:, 0:1]
            df.columns = ['review']
            df = df.dropna()

            # -------- CLEAN -------- #
            df['cleaned'] = df['review'].apply(clean_text)

            # -------- MODELS -------- #
            df['sentiment_vader'] = df['cleaned'].apply(
                lambda x: analyze_text(x)['label']
            )

            df['sentiment_ml'] = df['cleaned'].apply(predict_sentiment)

            df['match'] = df['sentiment_vader'] == df['sentiment_ml']

            # -------- KEYWORDS -------- #
            all_text = " ".join(df['cleaned'])
            keywords = extract_keywords(all_text)

        st.success("Analysis Completed ✅")

        # -------- METRICS -------- #
        pos = (df['sentiment_ml'] == "Positive").sum()
        neg = (df['sentiment_ml'] == "Negative").sum()
        neu = (df['sentiment_ml'] == "Neutral").sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", len(df))
        col2.metric("Positive", pos)
        col3.metric("Negative", neg)
        col4.metric("Neutral", neu)

        # -------- AGREEMENT -------- #
        st.subheader("📊 Model Agreement")
        agreement = df['match'].mean()
        st.write(f"Agreement: {agreement*100:.2f}%")

        # -------- DISAGREEMENT -------- #
        st.subheader("⚠️ Disagreement Cases")
        st.dataframe(
            df[df['match'] == False][
                ['review','sentiment_vader','sentiment_ml']
            ].head(10)
        )

        # -------- FILTER -------- #
        st.subheader("🔍 Filter Reviews")

        selected = st.selectbox(
            "Select Sentiment",
            ["All","Positive","Negative","Neutral"]
        )

        if selected != "All":
            filtered_df = df[df['sentiment_ml'] == selected]
        else:
            filtered_df = df

        st.dataframe(filtered_df.head(10))

        # -------- COMPARISON GRAPHS 🔥 -------- #
        st.subheader("📊 Model Comparison")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 🔵 VADER")
            vader_counts = df['sentiment_vader'].value_counts()
            st.bar_chart(vader_counts)

        with colB:
            st.markdown("### 🟢 ML Model")
            ml_counts = df['sentiment_ml'].value_counts()
            st.bar_chart(ml_counts)

        # -------- PIE CHART -------- #
        st.subheader("🥧 Pie Comparison")

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%')
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.pie(ml_counts, labels=ml_counts.index, autopct='%1.1f%%')
            st.pyplot(fig2)

        # -------- WORD CLOUD -------- #
        st.subheader("☁️ Word Cloud")

        wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        st.image(wc.to_array())

        # -------- KEYWORDS -------- #
        st.subheader("🔑 Keywords")
        st.write(keywords)

        # -------- SUMMARY -------- #
        st.subheader("🧠 Summary")

        total = len(df)

        if pos > neg:
            st.success(f"{(pos/total)*100:.1f}% customers satisfied")
        elif neg > pos:
            st.error(f"{(neg/total)*100:.1f}% negative feedback")
        else:
            st.warning("Mixed sentiment")

        # -------- DOWNLOAD -------- #
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Results",
            data=csv,   
            file_name="sentiment_analysis.csv",
            mime="text/csv"
        )