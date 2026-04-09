from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_text(text):
    if not text:
        return {"label":"Neutral","score":0}
    
    score = analyzer.polarity_scores(str(text))
    compound=score['compound']

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "score": compound
    }