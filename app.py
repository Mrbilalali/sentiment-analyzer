import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re

# -- Load the trained Keras model (GRU) --
@st.cache_resource  # cache the model so it loads only once
def load_sentiment_model():
    model = tf.keras.models.load_model('gru_model.h5')
    return model

model = load_sentiment_model()

# -- Load the tokenizer --
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# -- Text cleaning function (same as used during training) --
def clean_text(text):
    text = text.lower()                            # lowercase
    text = re.sub(r'<.*?>', '', text)             # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)          # remove punctuation and numbers
    return text

# -- Streamlit UI --
st.markdown("<h1 style='color:#00b36e;'>Huzaifa Analyzer</h1>", unsafe_allow_html=True)
st.write("Enter a movie review and the GRU model will predict its sentiment.")

review = st.text_area("Movie Review:", height=150)

if st.button("Predict Sentiment"):
    if not review.strip():
        st.write("Please enter some text to analyze.")
    else:
        # Preprocess the input
        cleaned = clean_text(review)  # apply cleaning steps
        seq = tokenizer.texts_to_sequences([cleaned])
        max_len = 200  # must match training max length
        padded = pad_sequences(seq, maxlen=max_len)
        
        # Model prediction
        preds = model.predict(padded)   # returns array of probabilities
        pred_proba = preds[0]
        classes = ['Negative', 'Neutral', 'Positive']
        pred_index = np.argmax(pred_proba)
        pred_label = classes[pred_index]
        
        # Display results
        st.markdown(f"**<span style='color:#00b36e;'>Predicted sentiment:</span>** {pred_label}", unsafe_allow_html=True)
        st.markdown("**Confidence Scores:**")
        for i, label in enumerate(classes):
            st.markdown(f"<span style='color:#00b36e;'>{label}:</span> {pred_proba[i]*100:.2f}%", unsafe_allow_html=True)
