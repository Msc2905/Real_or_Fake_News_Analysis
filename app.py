import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open("Decision_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
with open("Logistic_model.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def main():
    st.title("Real and Fake News Detection")

    # User input
    user_input = st.text_area("Enter a news article:", "")

    if st.button("Predict"):
        if user_input:
            # Preprocess user input
            user_input_vectorized = vectorizer.transform([user_input])

            # Make prediction
            prediction = model.predict(user_input_vectorized)

            # Display result
            if prediction == 0:
                st.write("Prediction: Fake News")
            else:
                st.write("Prediction: Real News")

if __name__ == "__main__":
    main()
