import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re

# -- Load the trained Keras model (GRU) --
@st.cache_resource
def load_sentiment_model():
    model = tf.keras.models.load_model('gru_model.h5')
    return model

model = load_sentiment_model()

# -- Load the tokenizer --
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# -- Text cleaning function --
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# -- UI Styling --
st.markdown("""
    <style>
        .main-title {
            color: #00b36e;
            font-size: 2.5em;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.2em;
            color: #333333;
            margin-bottom: 10px;
        }
        .result {
            font-size: 1.5em;
            font-weight: bold;
        }
        .positive { color: #00b36e; }
        .neutral { color: #007bff; }
        .negative { color: #ff4d4d; }
    </style>
""", unsafe_allow_html=True)

# -- Streamlit UI --
st.markdown("<div class='main-title'>Huzaifa Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Enter a movie review and the GRU model will predict its sentiment.</div>", unsafe_allow_html=True)

review = st.text_area("Movie Review:", height=150)

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned = clean_text(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        max_len = 200
        padded = pad_sequences(seq, maxlen=max_len)

        preds = model.predict(padded)
        pred_proba = preds[0]
        classes = ['Negative', 'Neutral', 'Positive']
        pred_index = np.argmax(pred_proba)
        pred_label = classes[pred_index]

        # Set color class for result
        color_class = {
            'Negative': 'negative',
            'Neutral': 'neutral',
            'Positive': 'positive'
        }[pred_label]

        # Display result
        st.markdown(f"<div class='result {color_class}'>Predicted Sentiment: {pred_label}</div>", unsafe_allow_html=True)

        st.markdown("<div class='sub-header'>Confidence Scores:</div>", unsafe_allow_html=True)
        for i, label in enumerate(classes):
            color = {
                'Negative': '#ff4d4d',
                'Neutral': '#007bff',
                'Positive': '#00b36e'
            }[label]
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{label}:</span> {pred_proba[i]*100:.2f}%", unsafe_allow_html=True)
