mport tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing import sequence

# =========================
# Load Model (only once)
# =========================
model = tf.keras.models.load_model("simple_rnn_imdb.h5")

# =========================
# Load Config + Word Index
# =========================
with open("config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]
max_features = config["max_features"]

with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)


# =========================
# Preprocessing Function
# =========================
def preprocess(text):
    words = text.lower().split()
    
    encoded = []
    for word in words:
        index = word_index.get(word, 2)  # unknown word = 2
        
        if index < max_features:
            encoded.append(index + 3)  # IMDB offset

    padded = sequence.pad_sequences([encoded], maxlen=max_len)
    return padded


# =========================
# Prediction Function
# =========================
def predict_sentiment(text):
    processed = preprocess(text)

    prediction = model.predict(processed, verbose=0)[0][0]

    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return {
        "sentiment": sentiment,
        "score": float(prediction),
        "confidence": float(confidence)
    }
