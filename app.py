import os
import re
import html
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_sentiment_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_review(review):
    review = html.unescape(review)
    review = re.sub(r'<.*?>', '', review)
    review = review.lower()
    review = re.sub(r'[^a-zA-Z0-9\s]', '', review)
    review = re.sub(r'\s+', ' ', review).strip()
    words = review.split()
    clean_words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(clean_words)


def predict_sentiment(text):
    cleaned = clean_review(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if score > 0.5 else "Negative"
    confidence = score if score > 0.5 else 1 - score
    return label, round(confidence * 100, 2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "").strip()
    if not review:
        return jsonify({"error": "Please enter a review."}), 400
    label, confidence = predict_sentiment(review)
    return jsonify({"sentiment": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)
