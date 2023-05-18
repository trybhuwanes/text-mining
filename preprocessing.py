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

# Membaca file datasrt
tweets_raw = pd.read_csv("tweets_raw.csv")

# Cetak 5 baris pertama dari dataset
print(tweets_raw.head())

# Cetak summary statistics
# print(tweets_raw.describe())

# We do not need first two columns. Let's drop them out.
tweets_raw.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)

# Drop duplicated rows
tweets_raw.drop_duplicates(inplace=True)

# Created at column's type should be datatime
tweets_raw["Created at"] = pd.to_datetime(tweets_raw["Created at"])

# Fill the missing values with unknown tag
tweets_raw["Location"].fillna("unknown", inplace=True)

print(tweets_raw.head())

# Pre-Processing Function 
def preprocess_text(text): 
    # get lowercase
    text = text.lower()
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove urls
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    # remove punctuation
    text = text.translate(text.maketrans('', '', string.punctuation))
    # strip whitespace
    text = text.strip()
    
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    words = [word for word in tokens if not word in stop_words]
    text = " ".join(word for word in words)
    
    # Remove non-alphabetic characters and keep the words contains three or more letters
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

tweets_raw["result_processed"] = tweets_raw["Content"].apply(preprocess_text)

# Print the first fifteen rows of Processed
print(tweets_raw[["result_processed"]].head(15))