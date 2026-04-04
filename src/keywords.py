from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords(text):
    keywords = kw_model.extract_keywords(text,top_n=3)
    return [kw[0] for kw in keywords ]