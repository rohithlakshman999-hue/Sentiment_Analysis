import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words=set(stopwords.words('english'))

def clean_text():
    if not text:
        return "" 
    
    text=str(text).lower()
    text=re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words=text.split()
    words = [word for word in words if word not in stop_words]
    text=" ".join(words)

    return text
