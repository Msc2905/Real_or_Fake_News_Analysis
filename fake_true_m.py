import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

data_fake = pd.read_csv('/content/drive/MyDrive/Fake.csv',encoding='latin-1', on_bad_lines='skip')
data_true = pd.read_csv('/content/drive/MyDrive/True.csv',encoding='latin-1', on_bad_lines='skip')
data_fake['class'] = 0
data_true['class'] = 1
data = pd.concat([data_true,data_fake], axis=0,ignore_index = True)
data.head()
import string
import re
def wordopt(text):
    text = text.lower() # lower case
    text = re.sub('\[.*?\]','',text) # remove anything with and within brackets
    text = re.sub('\\W',' ',text) # removes any character not a letter, digit, or underscore
    text = re.sub('https?://\S+|www\.\S+','',text) # removes any links starting with https
    text = re.sub('<.*?>+','', text) # removes anything with and within < >
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # removes any string with % in it
    text = re.sub('\n','',text) # remove next lines
    text = re.sub('\w*\d\w*','', text) # removes any string that contains atleast a digit with zero or more characters
    return text
    data['text'] = data['text'].apply(wordopt)
    
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
LR = LogisticRegression()
LR.fit(xv_train,y_train)

#website
import streamlit as st 
st.title('Fake News')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
  input_data = vector.transform([input_text])
  prediction = LR.predict(input_data)
  return prediction[0]

if input_text:
  pred = prediction(input_text)
  if pred == 1:
     st.write('The news is Fake')
  else:
     st.write('The news is True')
