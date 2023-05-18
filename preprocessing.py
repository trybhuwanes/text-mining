import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

# Membaca file datasrt
tweets_raw = pd.read_csv("tweets_raw.csv")

# Cetak 5 baris pertama dari dataset
print(tweets_raw.head())

# Cetak summary statistics
print(tweets_raw.describe())

def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Menghapus karakter khusus
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenisasi
    tokens = word_tokenize(text)

    # Menghapus stop words
    stop_words = set(stopwords.words('english'))  # Ganti dengan stop words bahasa yang sesuai
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Menggabungkan kembali token menjadi teks
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text