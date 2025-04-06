import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy model
nlp = spacy.load('en_core_web_md')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove punctuation (keep apostrophes for contractions)
    text = re.sub(r'[^\w\s\']', '', text)
    
    return text

def loc_fetch2(text):
    doc = nlp(text)
    
    location = None
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            location = ent.text
            break
    
    return location if location else "Location not found"

