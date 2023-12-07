import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
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

# Load the pre-trained model
model_json_file = "model_architecture.json"
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load the pre-trained model weights
model_weights_file = "model_weights.h5"
model.load_weights(model_weights_file)


# Load other necessary components like tokenizer and label encoder
# Make sure to save these components during training and load them here

def preprocess_text(text):
    # Your pre-processing steps here
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    pos_tags = pos_tag(tokens)
    # Add more pre-processing steps as needed
    return ' '.join(tokens)

def predict_fake_news(text):
    # Pre-process the input text
    processed_text = preprocess_text(text)
    
    # Tokenize and pad the sequence
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts([processed_text])
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=200, padding='post')

    # Make prediction
    prediction = model.predict(padded_seq)[0][0]

    return prediction

# Streamlit UI
st.title("Fake News Detection App")

user_input = st.text_area("Enter the news text:")
if st.button("Predict"):
    if user_input:
        result = predict_fake_news(user_input)
        st.write(f"The news has a {result*100:.2f}% chance of being fake.")
    else:
        st.warning("Please enter some news text.")
