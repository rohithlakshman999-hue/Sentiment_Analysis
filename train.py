import pandas as pd
import bz2
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ---------------- CLEAN ---------------- #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------- LOAD DATA ---------------- #
reviews = []
labels = []

print("Loading dataset...")

with bz2.open("data/test.ft.txt.bz2", "rt", encoding="utf-8") as file:
    for i, line in enumerate(file):

        if line.startswith("__label__1"):
            labels.append(0)
            reviews.append(line.replace("__label__1 ", "").strip())

        elif line.startswith("__label__2"):
            labels.append(1)
            reviews.append(line.replace("__label__2 ", "").strip())

# 🔥 USE FULL DATA (no break)


df = pd.DataFrame({
    "review": reviews,
    "label": labels
})

print("Total samples:", len(df))


# ---------------- CLEAN DATA ---------------- #
df['review'] = df['review'].apply(clean_text)

df = df[df['review'].str.len() > 20]   # remove short text
df = df.drop_duplicates()


# ---------------- SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.15, random_state=42
)


# ---------------- TF-IDF ---------------- #
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=30000,
    stop_words='english',
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ---------------- MODEL ---------------- #
model = LogisticRegression(
    max_iter=3000,
    C=2,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)


# ---------------- EVALUATION ---------------- #
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ---------------- SAVE ---------------- #
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel saved successfully ✅")