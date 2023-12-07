# Import necessary libraries
import streamlit as st
import pandas as pd
import tensorflow
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model architecture from JSON
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create the model from JSON
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights('model_weights.h5')

# Load tokenizer 
tokenizer = Tokenizer()

# Function to preprocess and predict
def predict_fake_news(title, text):
    input_text = title.lower() + ' ' + text.lower()
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=200, padding='post')
    prediction = loaded_model.predict(input_pad)[0][0]
    return prediction

# Streamlit app
def main():
    st.title("Fake News Detection App")

    # Input fields for user to enter title and text
    title = st.text_input("Enter the title:")
    text = st.text_area("Enter the text:")

    # Button to trigger prediction
    if st.button("Predict"):
        # Perform prediction
        prediction = predict_fake_news(title, text)

        # Display result
        st.write(f"Prediction: {prediction:.2%} chance of being fake news")

if __name__ == "__main__":
    main()
