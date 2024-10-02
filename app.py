from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the saved model
model = load_model('models/malaria_model.h5')  # Update the path if necessary

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

    if file and allowed_file(file.filename):
        # Secure the filename and generate a unique name to prevent overwriting
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        # Process the image for prediction
        img = Image.open(file_path)
        img = img.resize((img_height, img_width))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Reshape to match model input
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class]
        
        # Pass the filename and prediction to the result template
        return render_template('result.html', label=predicted_label, image_file=unique_filename)
    
    else:
        return 'Invalid file type.', 400

if __name__ == "__main__":
    app.run(debug=True)
