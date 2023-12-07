import streamlit as st
import pandas as pd
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
import pickle
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download("stopwords")

# Load the pre-trained model architecture from a JSON file
try:
    with open('model_architecture.json', 'r') as json_file:
        loaded_model_json = json_file.read()
except FileNotFoundError:
    st.error("Model architecture file not found. Please ensure it exists and is accessible.")
    exit()

# Load the pre-trained model weights
try:
    model_weights_file = 'model_weights.h5'
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_file)
except FileNotFoundError:
    st.error("Model weights file not found. Please ensure it exists and is accessible.")
    exit()

# Load the label encoder
try:
    with open('label_encoder.pkl', 'rb') as pickle_file:
        label_encoder = pickle.load(pickle_file)
except FileNotFoundError:
    st.error("Label encoder file not found. Please ensure it exists and is accessible.")
    exit()

# Load the tokenizer
try:
    with open('tokenizer.pkl', 'rb') as pickle_file:
        tokenizer = pickle.load(pickle_file)
except FileNotFoundError:
    st.error("Tokenizer file not found. Please ensure it exists and is accessible.")
    exit()

MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200

def preprocess_text(title, text):
    # Lowercase conversion and punctuation removal
    title = title.lower()
    text = text.lower()
    for char in string.punctuation:
        title = title.replace(char, "")
        text = text.replace(char, "")
    
    # Tokenization, stop word removal, and lemmatization
    title_tokens = word_tokenize(title)
    text_tokens = word_tokenize(text)
    title_tokens = [token for token in title_tokens if token not in nltk.corpus.stopwords.words('english')]
    text_tokens = [token for token in text_tokens if token not in nltk.corpus.stopwords.words('english')]
    title_tokens = [WordNetLemmatizer().lemmatize(token) for token in title_tokens]
    text_tokens = [WordNetLemmatizer().lemmatize(token) for token in text_tokens]
    
    # Add more pre-processing steps as needed
    
    return ' '.join(title_tokens + text_tokens)

def predict_fake_news(title, text):
    # Pre-process the input title and text
    processed_text = preprocess_text(title, text)
    
    # Tokenize and pad the sequence
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Make prediction
    prediction = model.predict(padded_seq)[0][0]
    
    # Convert prediction to label and map it back to its meaning
    label_index = round(prediction)
    label = label_encoder.inverse_transform([label_index])[0]

    # Return the label and prediction probability
    return label, prediction

# Streamlit UI
st.title("Fake News Detection App")

# Input text from the user
title_input = st.text_input("Enter the news title:")
text_input = st.text_area("Enter the news text:")

# Button to trigger prediction
if st.button("Predict"):
    if title_input and text_input:
        # Make prediction
        label, prediction = predict_fake_news(title_input, text_input)

        # Display the result
                # Display the result
        if label == "fake":
            st.write(f"The news is predicted as **fake** with a probability of {prediction * 100:.2f}%")
        else:
            st.write(f"The news is predicted as **real** with a probability of {prediction * 100:.2f}%")

