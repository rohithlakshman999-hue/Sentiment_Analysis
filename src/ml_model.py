import joblib

# Load trained model
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_sentiment(text):
    if not text:
        return "Neutral"

    text_vec = vectorizer.transform([text])

    # Get probabilities
    probs = model.predict_proba(text_vec)[0]
    confidence = max(probs)

    prediction = model.predict(text_vec)[0]

    # 🔥 Neutral logic
    if confidence < 0.60:
        return "Neutral"

    return "Positive" if prediction == 1 else "Negative"