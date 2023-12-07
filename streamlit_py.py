import streamlit as st
import tensorflow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
import nltk
import json

# Load your model architecture from a JSON file
with open('model_architecture.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load your model weights
model.load_weights('model_weights.h5')  # replace with your actual model weights file path
max_sequence_length = 100  # adjust based on your model's input shape

# Load tokenizer (assuming you saved it during training)
tokenizer = Tokenizer()
tokenizer.word_index = {'your': 1, 'token': 2, 'mapping': 3}  # replace with your actual tokenizer loading logic

# Other NLTK downloads and setups
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('punkt')

# Create a lemmatizer
lemmatizer = WordNetLemmatizer()

# Create a sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Remove common English stop words
stop_words = set(stopwords.words('english'))

# Streamlit app
st.title('Fake News Detection App')

# Text inputs for news title and news text
news_title = st.text_area('Enter the news title:', '')
news_text = st.text_area('Enter the news text:', '')

# Preprocess and make predictions when a button is clicked
if st.button('Predict'):
    try:
        # Preprocess the input text
        news_title = news_title.lower()
        news_text = news_text.lower()

        # Remove punctuation
        news_title = ''.join(ch for ch in news_title if ch not in string.punctuation)
        news_text = ''.join(ch for ch in news_text if ch not in string.punctuation)

        # Tokenize and lemmatize
        title_tokens = word_tokenize(news_title)
        title_tokens = [token for token in title_tokens if token not in stop_words]
        title_tokens = [lemmatizer.lemmatize(token) for token in title_tokens]

        text_tokens = word_tokenize(news_text)
        text_tokens = [token for token in text_tokens if token not in stop_words]
        text_tokens = [lemmatizer.lemmatize(token) for token in text_tokens]

        # Combine title and text tokens
        tokens = title_tokens + text_tokens

        # POS tagging
        pos_tags = pos_tag(tokens)

        # Sentiment analysis
        title_sentiment = analyzer.polarity_scores(news_title)
        text_sentiment = analyzer.polarity_scores(news_text)

        # Convert tokens to sequences using the tokenizer
        sequences = tokenizer.texts_to_sequences([tokens])

        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

        # Make predictions using the loaded model
        prediction = model.predict(padded_sequences)

        # Assuming a binary classification, you might display the result like this
        st.write(f'Prediction: {prediction[0][0]}')
        st.write(f'Title Tokens: {title_tokens}')
        st.write(f'Text Tokens: {text_tokens}')
        st.write(f'POS Tags: {pos_tags}')
        st.write(f'Title Sentiment Scores: {title_sentiment}')
        st.write(f'Text Sentiment Scores: {text_sentiment}')

    except Exception as e:
        st.error(f'Error: {str(e)}')
