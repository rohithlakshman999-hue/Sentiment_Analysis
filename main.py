import pandas as pd 
from src.preprocess import clean_text
from src.keywords import extract_keywords
from src.sentiment import analyze_text

df = pd.read_csv("data/review.csv")

df = df.head(10000)

df = df.dropna(subset=[['review']])

df['cleaned'] = df['review'].apply(clean_text)

df['sentiment'] = df['cleaned'].apply(lambda x: analyze_text(x)['label'])
df['sentiment'] = df['cleaned'].apply(lambda x: analyze_text(x)['score'])

df['keywords'] = df['cleaned'].apply(lambda x: extract_keywords(x))

df.to_csv("data/output.csv", index=False)

print("Analysis Completed ✅")