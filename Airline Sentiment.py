#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline



# In[86]:


df = pd.read_csv('tweets.csv')


# In[87]:


df.head()


# In[88]:


# checking for missing values
df.isnull().sum()


# In[89]:


# checking if it as any duplicates
df.duplicated()


# In[90]:


import re

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)
    # Remove special characters and punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text


# In[91]:


df['clean_text'] = df['text'].apply(preprocess_text)



# In[92]:


# Classify sentiments
def classify_sentiment(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return -1
    else:
        return 0
    


# In[94]:


# Group by sentiment and count the number of tweets
sentiment_counts = df['airline_sentiment'].value_counts().reset_index()

# Rename the columns for clarity
sentiment_counts.columns = ['Sentimnet', 'Count']

#Display the summary table
sentiment_counts.style.background_gradient(cmap='coolwarm_r')


# In[95]:


df['sentiment_class'] = df['airline_sentiment'].apply(classify_sentiment)


# In[96]:


#train-test split
X_train, X_test, Y_train, Y_test = train_test_split(df['clean_text'], df['sentiment_class'], test_size=0.2, random_state=42)


# In[97]:


#define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('clf', LogisticRegression(max_iter=1000))
])


# In[98]:


#Train model
pipeline.fit(X_train, Y_train)


# In[99]:


y_pred = pipeline.predict(X_test)


# In[100]:


# model evaluation
accuracy=accuracy_score(Y_test, y_pred)
print("accuracy", accuracy)
print(classification_report(Y_test, y_pred))


# In[ ]:




