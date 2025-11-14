import os
import requests  # Import requests
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from ultralytics import YOLO
from PIL import Image

# --- NEW: Code to download the model ---
MODEL_URL = "https://github.com/GuruAditya/pothole_detection/releases/download/V1/best.2.pt"  # <-- PASTE YOUR LINK
MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    print("Model not found, downloading...")
    os.makedirs("models", exist_ok=True)
    try:
        r = requests.get(MODEL_URL, allow_redirects=True, stream=True)
        r.raise_for_status()  # Check for errors
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("...model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
# --- End of new code ---

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load your model (this now loads the downloaded file)
print("Loading YOLO model...")
model = YOLO(MODEL_PATH) 
print("...model loaded.")

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and runs YOLO inference."""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run YOLO inference
        results = model(filepath)
        
        # Save the result image (this will have the boxes drawn)
        # Assumes 'results' is a list and we take the first one
        result_img_array = results[0].plot() # .plot() returns a numpy array (BGR)
        result_img = Image.fromarray(result_img_array[..., ::-1]) # Convert BGR to RGB

        result_filename = 'result_' + filename
        result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_img.save(result_filepath)

        # Pass the *path* of the result image to the template
        return render_template('result.html', result_image=result_filename)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False for production