import pandas as pd
import bz2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------- LOAD DATA ---------------- #
reviews = []
labels = []

with bz2.open("data/test.ft.txt.bz2", "rt", encoding="utf-8") as file:
    for i, line in enumerate(file):

        # Negative
        if line.startswith("__label__1"):
            labels.append(0)
            reviews.append(line.replace("__label__1 ", "").strip())

        # Positive
        elif line.startswith("__label__2"):
            labels.append(1)
            reviews.append(line.replace("__label__2 ", "").strip())

        # Limit dataset for faster training
        if i >= 50000:
            break

# Convert to DataFrame
df = pd.DataFrame({
    "review": reviews,
    "label": labels
})

# ---------------- CHECK LABELS ---------------- #
print("Labels found:", set(labels))

if len(set(labels)) < 2:
    raise ValueError("Dataset must contain at least 2 classes!")

# ---------------- SPLIT DATA ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42
)

# ---------------- TF-IDF (IMPROVED) ---------------- #
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # already good
    max_features=10000,      # increase
    stop_words='english',
    min_df=5,                # ignore rare words
    max_df=0.8               # ignore very common words
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- MODEL ---------------- #
model = LogisticRegression(
    max_iter=2000,
    C=2,              # stronger learning
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

# ---------------- EVALUATION ---------------- #
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ---------------- #
import os

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel saved successfully ✅")