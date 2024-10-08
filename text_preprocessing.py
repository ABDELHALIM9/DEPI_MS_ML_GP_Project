import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# Download stopwords and lemmatizer resources from nltk
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# Preprocessing function
def preprocess_text(text):
    # Making text lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove Mentions and Hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove Special Characters and Punctuation
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    
    # Remove Numbers
    text = re.sub(r'\d+', '', text)
    
    # Expand contractions
    text = expand_contractions(text)
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    # Handle emoticons and emojis
    text = emoji.demojize(text)
    
    # Reduce repeated characters (e.g., "soooo" becomes "soo")
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    return text
    