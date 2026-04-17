import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing import sequence

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("simple_rnn_imdb.h5", compile=False)

model = load_model()

# =========================
# Load Files
# =========================
@st.cache_data
def load_files():
    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    with open("word_index.pkl", "rb") as f:
        word_index = pickle.load(f)

    with open("reverse_word_index.pkl", "rb") as f:
        reverse_word_index = pickle.load(f)

    return config, word_index, reverse_word_index


config, word_index, reverse_word_index = load_files()

max_len = config["max_len"]
max_features = config["max_features"]

# =========================
# Preprocessing
# =========================
def preprocess_text(text):
    words = text.lower().split()

    encoded = []
    for word in words:
        index = word_index.get(word, 2)
        if index < max_features:
            encoded.append(index + 3)

    padded = sequence.pad_sequences([encoded], maxlen=max_len)
    return padded, encoded


# =========================
# Decode Function
# =========================
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review and find out whether it's **positive or negative**.")

user_input = st.text_area("✍️ Enter your review here:", height=150)

# =========================
# Button Logic
# =========================
if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before clicking analyze.")

    else:
        processed_input, encoded = preprocess_text(user_input)

        prediction = model.predict(processed_input, verbose=0)[0][0]

        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.subheader("📊 Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Score:** {confidence:.4f}")

        st.progress(float(prediction))

        st.subheader("🔍 Processed Text (Model Input)")
        decoded = decode_review(encoded)
        st.write(decoded)

else:
    st.info("Enter a review above and click 'Analyze Sentiment'.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
