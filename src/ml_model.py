import joblib

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_sentiment(text):
    if not text:
        return "Neutral"

    text_vec = vectorizer.transform([text])

    probs = model.predict_proba(text_vec)[0]
    confidence = max(probs)

    prediction = model.predict(text_vec)[0]

    text_lower = text.lower()

    # 🔥 smart rule for negatives
    if "not" in text_lower:
        return "Negative"

    # 🔥 improved neutral logic
    if confidence < 0.55:
        return "Neutral"

    return "Positive" if prediction == 1 else "Negative"