import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import emoji

# Load the trained model
def create_model():
    model = tf.keras.Sequential([
        Embedding(input_dim=20000, output_dim=128),
        Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.load_weights('sentiment_model_tune.h5')

# Tokenizer setup (recreate or load tokenizer used during training)
tokenizer = Tokenizer(num_words=5000)

# Define preprocessing functions
def handle_negations(text):
    negation_patterns = ["n't", "not", "never", "no"]
    words = text.split()
    for i in range(len(words)):
        if words[i].lower() in negation_patterns:
            if i+1 < len(words):
                words[i+1] = "NOT_" + words[i+1]
    return ' '.join(words)

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'\w+:\/\/\S+', ' ', text)
        text = re.sub(r'\S*@\S*\s?', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[^a-zA-Z]+', ' ', text)
        text = handle_negations(text)
        text = emoji.demojize(text)
        text = re.sub(r'\d+', ' ', text)
        text = text.lower()
    return text

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(tweet):
    tokens = word_tokenize(tweet)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tweet = ' '.join(tokens)
    return tweet

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    preprocessed_text = preprocess_text(cleaned_text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=30)
    prediction = model.predict(padded_sequence)
    return prediction

# Streamlit app
st.title("Sentiment Analysis")

input_text = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    prediction = predict_sentiment(input_text)
    sentiment = np.argmax(prediction)
    sentiments = ['Negative', 'Neutral', 'Positive']
    st.write(f"Predicted Sentiment: {sentiments[sentiment]}")
