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
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download("stopwords")

# Load the pre-trained model architecture from a JSON file
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the pre-trained model weights
model_weights_file = 'model_weights.h5'

# Load the pre-trained model
model = model_from_json(loaded_model_json)
model.load_weights(model_weights_file)

# Load other necessary components like tokenizer and label encoder
# Make sure to save these components during training and load them here
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts([""])  # Empty fit to avoid errors

def preprocess_text(title, text):
    # Your pre-processing steps here
    title = title.lower()
    text = text.lower()
    
    title = ''.join(ch for ch in title if ch not in string.punctuation)
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    
    title_tokens = word_tokenize(title)
    text_tokens = word_tokenize(text)
    
    title_tokens = [token for token in title_tokens if token not in nltk.corpus.stopwords.words('english')]
    text_tokens = [token for token in text_tokens if token not in nltk.corpus.stopwords.words('english')]
    
    title_tokens = [WordNetLemmatizer().lemmatize(token) for token in title_tokens]
    text_tokens = [WordNetLemmatizer().lemmatize(token) for token in text_tokens]
    
    title_pos_tags = pos_tag(title_tokens)
    text_pos_tags = pos_tag(text_tokens)
    
    # Add more pre-processing steps as needed
    
    return ' '.join(title_tokens + text_tokens)

def predict_fake_news(title, text):
    # Pre-process the input title and text
    processed_text = preprocess_text(title, text)
    
    # Tokenize and pad the sequence
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=200, padding='post')

    # Make prediction
    prediction = model.predict(padded_seq)[0][0]

    return prediction

# Streamlit UI
st.title("Fake News Detection App")

# Input text from the user
title_input = st.text_input("Enter the news title:")
text_input = st.text_area("Enter the news text:")

# Button to trigger prediction
if st.button("Predict"):
    if title_input and text_input:
        # Make prediction
        prediction = predict_fake_news(title_input, text_input)

        # Display the result
        st.write(f"The news has a {prediction * 100:.2f}% chance of being fake.")
    else:
        st.warning("Please enter both news title and text.")
