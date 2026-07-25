import streamlit as st
import tensorflow as tf
import pickle
import re
import html
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences


nltk.download('stopwords')

# Load Model & Tokenizer

MODEL_PATH = "cnn_sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200  


# Text Cleaning Function

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_review(review):
    review = html.unescape(review)
    review = re.sub(r'<.*?>', '', review)
    review = review.lower()
    review = re.sub(r'[^a-zA-Z0-9\s]', '', review)
    review = re.sub(r'\s+', ' ', review)
    review = review.strip()

    words = review.split()
    clean_words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(clean_words)


# Prediction Function

def predict_sentiment(text):
    cleaned_text = clean_review(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(padded_sequence, verbose=0)[0][0]

    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"

    return sentiment, float(prediction)

# Streamlit UI

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review and the CNN model will predict the sentiment."
)

review = st.text_area(
    "Movie Review",
    height=150,
    placeholder="Type your review here..."
)

if st.button("Predict Sentiment"):
    if review.strip():
        sentiment, score = predict_sentiment(review)

        st.subheader("Prediction")

        if "Positive" in sentiment:
            st.success(sentiment)
        else:
            st.error(sentiment)

        st.write(f"**Confidence Score:** {score:.4f}")

    else:
        st.warning("Please enter a review.")
