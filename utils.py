import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources only once
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')  
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Run the function once when utils.py is imported
download_nltk_resources()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)
