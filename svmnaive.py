#import library yang dibutuhkan
import pandas as pd 
import re 
import string 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk.corpus  
from nltk.text import Text 
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca file dataset
tweets_raw = pd.read_csv("tweets_april_raw.csv")

# Cetak 5 baris pertama dari dataset
print(tweets_raw.head())

# # Menghapus kolom yang tidak diperlukan
# deleteColumns = ["coordinates", "hashtags", "media", "urls", "in_reply_to_screen_name", "in_reply_to_status_id", "in_reply_to_user_id", "place", "quote_id", "retweet_count", "retweet_id", "retweet_screen_name", "source", "tweet_url", "user_id", "user_default_profile_image", "user_description", "user_favourites_count", "user_followers_count", "user_friends_count", "user_listed_count", "user_location", "user_name", "user_screen_name", "user_statuses_count", "user_time_zone", "user_urls", "user_verified"]
tweets_raw.drop(columns=["coordinates", "hashtags", "media", "urls", "favorite_count", "id", "in_reply_to_screen_name", "in_reply_to_status_id", "in_reply_to_user_id", "lang", "place", "quote_id", "retweet_count", "retweet_id", "retweet_screen_name", "source", "tweet_url","user_created_at", "user_id", "user_default_profile_image", "user_description", "user_favourites_count", "user_followers_count", "user_friends_count", "user_listed_count", "user_location", "user_name", "user_screen_name", "user_statuses_count", "user_time_zone", "user_urls", "user_verified"], axis=1, inplace=True)

# # Menghapus baris yang ada duplikasinya
tweets_raw.drop_duplicates(inplace=True)

# # Membuat kolom created at
# tweets_raw["Created at"] = pd.to_datetime(tweets_raw["Created at"])

# # MEngisi nilai yang kosong
tweets_raw["possibly_sensitive"].fillna("TRUE", inplace=True)

print(tweets_raw.head())
print(tweets_raw.isna().sum())

# Histogram kata-kata paling sering muncul
word_freq = tweets_raw["text"].str.split().explode().value_counts().reset_index()
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
    text = re.sub(r'^https?:\/\/.[\r\n]', '', text)
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
tweets_raw["result_processed"] = tweets_raw["text"].apply(preprocess_text)

# Mencetak hasil preprocessing sebanyak 14 data
print(tweets_raw[["result_processed"]].head(15))

# Mengubah 
tweets_raw['possibly_sensitive'] = tweets_raw['possibly_sensitive'].map({'TRUE':1,False:0})
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(tweets_raw['result_processed'],tweets_raw['possibly_sensitive'],test_size=0.2, random_state=100)

Encoder = LabelEncoder()
Y_train = Encoder.fit_transform(Y_train)
Y_test = Encoder.fit_transform(Y_test)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(tweets_raw['result_processed'])
X_train_Tfidf = Tfidf_vect.transform(X_train)
X_test_Tfidf = Tfidf_vect.transform(X_test)
print(Tfidf_vect.vocabulary_)

print(X_train_Tfidf)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_Tfidf,Y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y_test)*100)

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train_Tfidf,Y_train)
# predict the labels on validation dataset
predictions_NB = Naive.predict(X_test_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Y_test)*100)