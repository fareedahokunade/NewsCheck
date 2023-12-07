import Flask
import tensorflow

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model architecture from JSON
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create model from JSON
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights('model_weights.h5')

# Load tokenizer
# Note: You may need to adjust this based on your preprocessing steps
tokenizer = Tokenizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']

    # Preprocess the input (adjust based on your preprocessing steps)
    input_text = title.lower() + ' ' + text.lower()
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_pad = pad_sequences(input_seq, maxlen=200, padding='post')

    # Make predictions
    prediction = loaded_model.predict(input_pad)[0][0]

    # Return the result
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
