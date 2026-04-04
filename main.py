import pandas as pd
from src.preprocess import clean_text
from src.sentiment import analyze_sentiment
from src.keywords import extract_keywords

df = pd.read_csv ("data/review.csv")

df = df.dropna(subset=['review'])

df['cleaned'] = df['review'].apply(clean_text)

df['sentiment'] = df['cleaned'].apply(analyze_sentiment)

df['keyboard'] = df['cleaned'].apply(lambda x: extract_keywords(x))


df.to_csv("data/output.csv", index=False)

print("Analyzes Completed !")