# import libraries
import pandas as pd
import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# Load your data frame samples
fake_path = r"E:\1 ExaleR\Project\264\Fake.csv"
fake_sample = pd.read_csv(fake_path, encoding="latin1", on_bad_lines='skip')

true_path = r"E:\1 ExaleR\Project\264\Fake.csv"
true_sample = pd.read_csv(true_path, encoding="latin1", on_bad_lines='skip')

# Load the SVM model
with open(r"E:\1 ExaleR\Project\264\Logistic_model", 'rb') as model_file:
    svc = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open(r"E:\1 ExaleR\Project\264\Logistic_model", 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Create a function to clean text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Prediction function
def news_prediction(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    new_x_test = new_def_test['text']
    new_tfidf_test = tfidf_vectorizer.transform(new_x_test).toarray()  # Convert to dense array
    pred_dt = svc.predict(new_tfidf_test)

    if pred_dt[0] == 0:
        return "This is Fake News!"
    else:
        return "The News seems to be True!"

def main():
    # Write the app title and introduction
    st.title("Fake News Prediction System")
    st.write("Context: ... (your description)")

    # User input area
    user_text = st.text_area('Text to Analyze', '''(paste news text here)''', height=350)
    
    # Button to trigger analysis
    if st.button("Article Analysis Result"):
        news_pred = news_prediction(user_text)
        if news_pred == "This is Fake News!":
            st.error(news_pred, icon="🚨")
        else:
            st.success(news_pred)
            st.balloons()

    # Sample articles section
    st.write("## Sample Articles to Try:")
    st.write("#### Fake News Article")
    st.write("Click the box below and copy/paste.")
    st.dataframe(fake_sample['text'].sample(1), hide_index=True)

    st.write("#### Real News Article")
    st.write("Click the box below and copy/paste.")
    st.dataframe(true_sample['text'].sample(1), hide_index=True)

if __name__ == "__main__":
    main()
