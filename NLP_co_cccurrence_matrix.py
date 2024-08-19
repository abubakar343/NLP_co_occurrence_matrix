#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import WordNetLemmatizer  # To perform lemmatization
from sklearn.feature_extraction.text import CountVectorizer  # To perform Bag of Words (BoW) representation
from sklearn.feature_extraction.text import TfidfVectorizer  # To perform TF-IDF representation
from nltk import pos_tag  # For parts of speech tagging
from sklearn.metrics import pairwise_distances  # To perform cosine similarity
from nltk import word_tokenize  # To create tokens
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import wordpunct_tokenize

# Downloading NLTK resources
nltk.download('stopwords') 
nltk.download('punkt') 
nltk.download('wordnet')

# Loading stopwords for text preprocessing
stop = stopwords.words('english')

# Sample text for processing
df_1 = ["It is going to rain today. Today I am not going outside. NLP is an interesting topic. NLP includes ML, DL topics too. I am going to complete NLP homework, today"]

# Bag of Words (BoW) representation
vector = CountVectorizer()
data = vector.fit_transform(df_1)  # Transforming the text data into a BoW matrix
vocab = vector.get_feature_names()  # Getting the list of features (words)
lenght = len(vocab)  # Number of unique words
matrix = np.zeros((lenght, lenght))  # Initializing a matrix to hold word co-occurrence counts
matrix = pd.DataFrame(matrix)  # Converting the matrix to a DataFrame for easier manipulation
matrix

# Creating a word co-occurrence matrix
window_length = 3  # Context window size for counting co-occurrences
for index, word in enumerate(vocab):
    for context_index in range((max(0, index - window_length)), (min(len(vocab), index + window_length + 1))):
        if vocab[context_index] in vocab:
            row_index = vocab.index(word)
            col_index = vocab.index(vocab[context_index])
            matrix[row_index][col_index] += 1  # Updating the co-occurrence count

# Computing cosine similarity between words
from sklearn.metrics.pairwise import cosine_similarity
cosine = cosine_similarity(matrix)  # Computing cosine similarity between rows of the matrix
print(cosine)

# Sample text for tokenization and cleaning
df = "It is going to rain today. Today I am not going outside. NLP is an interesting topic. NLP includes ML, DL topics too. I am going to complete NLP homework, today"

# Removing special characters and tokenizing the text
spl_char_text = re.sub(r'[^ a-z]', '', df)  # Removing non-alphabetic characters
tokens = wordpunct_tokenize(spl_char_text)  # Tokenizing the cleaned text
tokens

# TF-IDF representation
df = ["It is going to rain today", "Today I am not going outside", "NLP is an interesting topic", "NLP includes ML, DL topics too", "I am going to complete NLP homework, today"]
vectorizer = TfidfVectorizer()  # Initializing the TF-IDF vectorizer
matrix = vectorizer.fit_transform(df)  # Transforming the text data into a TF-IDF matrix
print(matrix)

# Computing cosine similarity between TF-IDF vectors
from sklearn.metrics.pairwise import cosine_similarity
cosine = cosine_similarity(matrix)  # Computing cosine similarity between rows of the TF-IDF matrix
print(cosine)
