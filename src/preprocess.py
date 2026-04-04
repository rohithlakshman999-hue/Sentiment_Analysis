import re

def clean_text(text):
    text= str(text).lower()
    text=re.sub(r'[^\w\s]','',text)
    return text