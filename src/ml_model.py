import joblib

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    return "Positive" if prediction == 1 else "Negative"