import pandas as pd 
from src.preprocess import clean_text
from src.keywords import extract_keywords
from src.sentiment import analyze_text
from src.ml_model import predict_sentiment

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("data/review.csv")

# limit rows
df = df.head(10000)

# remove empty rows
df = df.dropna(subset=['review'])

# ---------------- CLEAN TEXT ---------------- #
df['cleaned'] = df['review'].apply(clean_text)

# ---------------- VADER MODEL ---------------- #
df['sentiment_vader'] = df['cleaned'].apply(
    lambda x: analyze_text(x)['label']
)

df['score_vader'] = df['cleaned'].apply(
    lambda x: analyze_text(x)['score']
)

# ---------------- YOUR ML MODEL ---------------- #
df['sentiment_ml'] = df['cleaned'].apply(predict_sentiment)

# ---------------- KEYWORDS (FAST WAY) ---------------- #
all_text = " ".join(df['cleaned'])
keywords = extract_keywords(all_text)

# store same keywords for reference (optional)
df['keywords'] = str(keywords)

# ---------------- SAVE OUTPUT ---------------- #
df.to_csv("data/output.csv", index=False)

print("Analysis Completed ✅")