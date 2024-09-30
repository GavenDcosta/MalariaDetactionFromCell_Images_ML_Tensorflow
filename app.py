from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = load_model('models/malaria_model.h5')  # Load your saved model

# Image dimensions
img_height, img_width = 100, 100  # Based on your model input size

# Define the classes
class_names = ['Parasitized', 'Uninfected']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400

    if file:
        img = Image.open(file)
        img = img.resize((img_height, img_width))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Reshape to match model input
        
        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class]

        return render_template('result.html', label=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)
