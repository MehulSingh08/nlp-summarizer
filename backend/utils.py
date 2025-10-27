import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Clean and preprocess text for TextRank summarization.
    
    Args:
        text (str): Raw input text
        
    Returns:
        list: List of cleaned sentences
    """
    if not text or not isinstance(text, str):
        return []
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    cleaned_sentences = []
    
    for sentence in sentences:
        # Skip very short sentences (less than 10 characters)
        if len(sentence.strip()) < 10:
            continue
            
        # Convert to lowercase
        sentence = sentence.lower()
        
        # Remove extra whitespace and newlines
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # Tokenize into words
        words = word_tokenize(sentence)
        
        # Remove punctuation and stopwords, keep only alphabetic words
        filtered_words = [
            word for word in words 
            if word.isalpha() and word not in stop_words and len(word) > 2
        ]
        
        # Only keep sentences with at least 3 meaningful words
        if len(filtered_words) >= 3:
            cleaned_sentence = ' '.join(filtered_words)
            cleaned_sentences.append(cleaned_sentence)
    
    return cleaned_sentences

def get_original_sentences(text):
    """
    Get original sentences for final summary display.
    
    Args:
        text (str): Raw input text
        
    Returns:
        list: List of original sentences
    """
    if not text or not isinstance(text, str):
        return []
    
    sentences = sent_tokenize(text)
    # Filter out very short sentences
    return [s.strip() for s in sentences if len(s.strip()) > 10]