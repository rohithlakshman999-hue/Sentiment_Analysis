import joblib

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_sentiment(text):
    if not text:
        return "Neutral"

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    # simple neutral logic
    if len(text.split()) < 3:
        return "Neutral"

    return "Positive" if prediction == 1 else "Negative"