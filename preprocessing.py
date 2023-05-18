import pandas as pd 
import re 
import string 
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk.corpus  
from nltk.text import Text 
from PIL import Image

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca file dataset
tweets_raw = pd.read_csv("tweets_raw.csv")

# Cetak 5 baris pertama dari dataset
print(tweets_raw.head())

# Menghapus kolom yang tidak diperlukan
tweets_raw.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)

# Menghapus baris yang ada duplikasinya
tweets_raw.drop_duplicates(inplace=True)

# Membuat kolom created at
tweets_raw["Created at"] = pd.to_datetime(tweets_raw["Created at"])

# MEngisi nilai yang kosong
tweets_raw["Location"].fillna("unknown", inplace=True)

print(tweets_raw.head())

# Histogram kata-kata paling sering muncul
word_freq = tweets_raw["Content"].str.split().explode().value_counts().reset_index()
word_freq.columns = ["Word", "Frequency"]

plt.figure(figsize=(12, 6))
sns.barplot(x="Frequency", y="Word", data= word_freq.head(20))
plt.title("20 Kata Paling Sering Muncul")
plt.xlabel("Frekuensi")
plt.ylabel("Kata")
plt.show()

# Fungsi Pre-Processing 
def preprocess_text(text): 
    # Mengganti ke huruf kecil
    text = text.lower()
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    # Menghapus URL/link
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    # Menghapus punctuation
    text = text.translate(text.maketrans('', '', string.punctuation))
    # Menghapus whitespace
    text = text.strip()
    # Stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    words = [word for word in tokens if not word in stop_words]
    text = " ".join(word for word in words)
    
    # Menghapus non-alphabetic dan tidak menghapus kata yang terdiri dari minimal 3 huruf
    tokens = word_tokenize(text)
    words = [word for word in
             tokens if word.isalpha() and len(word)>2]
    text = " ".join(word for word in words)
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    words = [lemmatizer.lemmatize(word, pos = 'a') for word in tokens]
    result = " ".join(word for word in words)
    return result

# Mengaplikasikan fungsi ke kolom Content dan disimpan di kolom result_preprocessed
tweets_raw["result_processed"] = tweets_raw["Content"].apply(preprocess_text)

# Mencetak hasil preprocessing sebanyak 14 data
print(tweets_raw[["result_processed"]].head(15))