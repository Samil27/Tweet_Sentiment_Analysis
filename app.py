import streamlit as st
import tensorflow as tf
import numpy as np
import re
import emoji
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your model
model_path = 'sentiment_model_tune.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess text
def preprocess_text(text):
    def handle_negations(text):
        negation_patterns = ["n't", "not", "never", "no"]
        words = text.split()
        for i in range(len(words)):
            if words[i].lower() in negation_patterns:
                if i + 1 < len(words):
                    words[i + 1] = "NOT_" + words[i + 1]
        return ' '.join(words)

    if isinstance(text, str):
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'\w+://\S+', ' ', text)
        text = re.sub(r'\S*@\S*\s?', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = handle_negations(text)
        text = emoji.demojize(text)
        text = re.sub(r'\d+', ' ', text)
        text = text.lower()
    return text

# Load tokenizer
tokenizer = Tokenizer(num_words=5000)
# Fit tokenizer on some dummy data (or load a pre-fitted tokenizer)
dummy_data = ["This is some example data.", "More example text."]
tokenizer.fit_on_texts(dummy_data)

# Function to predict sentiment from text input
def predict_sentiment(text):
    # Preprocess the input text
    text = preprocess_text(text)
    # Tokenize and pad the sequence
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=30)
    # Make prediction
    prediction = model.predict(padded_sequence)
    return prediction

# Streamlit UI
def main():
    st.title('Sentiment Analysis with Bidirectional LSTM')
    st.markdown('Enter your text to predict sentiment:')

    # User input text area
    user_input = st.text_area('Input Text', '')

    # Predict button
    if st.button('Predict'):
        if user_input.strip() == '':
            st.warning('Please enter some text.')
        else:
            # Predict sentiment
            prediction = predict_sentiment(user_input)
            # Display prediction result
            sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            predicted_sentiment = sentiment_mapping[np.argmax(prediction)]
            st.success(f'Predicted Sentiment: {predicted_sentiment}')

if __name__ == '__main__':
    main()
