#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import nltk 
import numpy as np
import re
from nltk.stem import wordnet                                  # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer    # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer    # to perform tfidf
from nltk import pos_tag                                       # for parts of speech
from sklearn.metrics import pairwise_distances                 # to perfrom cosine similarity
from nltk import word_tokenize                                 # to create tokens
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import wordpunct_tokenize


# In[ ]:


nltk.download('stopwords') 
nltk.download('punkt') 
nltk.download('wordnet')
stop = stopwords.words('english')


# In[ ]:


df_1 = ["It is going to rain today.Today I am not going outside. NLP is an interesting topic.NLP includes ML, DL topics too.I am going to complete NLP homework, today"]


# In[ ]:


vector = CountVectorizer()
data = vector.fit_transform(df_1)
vocab = vector.get_feature_names()
lenght = len(vocab)
matrix = np.zeros((lenght,lenght))
matrix = pd.DataFrame(matrix)
matrix


# In[ ]:


window_lenght = 3
for index,word in enumerate(vocab):
       for context_index in range((max(0,index - window_lenght)),(min(len(vocab),index + window_lenght +1))): 
            if vocab[context_index] in vocab:
                    row_index = vocab.index(word)
                    col_index = vocab.index(vocab[context_index])
                    matrix[row_index][col_index] += 1 


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
cosine = cosine_similarity(matrix)
print (cosine)


# In[ ]:


df = "It is going to rain today. Today I am not going outside.NLP is an interesting topic.NLP includes ML, DL topics too.I am going to complete NLP homework, today"


# In[ ]:


import re
spl_char_text = re.sub(r'[^ a-z]','',df) 
tokens = wordpunct_tokenize(spl_char_text)
tokens


# In[ ]:


df = ["It is going to rain today","Today I am not going outside","NLP is an interesting topic","NLP includes ML, DL topics too","I am going to complete NLP homework, today"]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df)
print(matrix)


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
cosine = cosine_similarity(matrix)
print (cosine)

