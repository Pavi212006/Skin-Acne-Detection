import os
import uuid # For generating unique filenames
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Configuration Variables ---

MODEL_PATH = 'acne_cnn_model.h5' 
UPLOAD_FOLDER = 'static/uploads'

# CRITICAL FIX: Image Size based on model.summary()
IMG_HEIGHT = 126   
IMG_WIDTH = 126    
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)

CLASSES = ['No Acne', 'Mild Acne', 'Moderate Acne', 'Severe Acne']

# ImageNet means and standard deviations for Z-score normalization
# This is used as the most likely pre-processing used during training.
IMAGENET_MEANS = np.array([0.485, 0.456, 0.406])
IMAGENET_STDS = np.array([0.229, 0.224, 0.225])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model variable
model = None

# --- Model Loading ---
try:
    # FIX: Use compile=False and ensure model._make_predict_function() is removed
    model = load_model(MODEL_PATH, compile=False) 
    print(f"CNN Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    # If loading fails, print the error to the console (where the true issue lies)
    print(f"FATAL Error loading model: {e}")
    model = None

# --- Image Preprocessing Function ---

def prepare_image(img_path):
    if not os.path.exists(img_path):
        return None

    try:
        img = Image.open(img_path).convert('RGB')
        
        # 1. Aspect Ratio Preservation (Padding to handle any shape/size)
        width, height = img.size
        max_dim = max(width, height)
        new_img = Image.new(img.mode, (max_dim, max_dim), color='black') 
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        new_img.paste(img, (left, top))
        
        # 2. Resize to model's required input size (126x126)
        img_resized = new_img.resize(TARGET_SIZE)

        img_array = image.img_to_array(img_resized) # Range [0, 255]
        
        # 3. Rescale pixel values from [0, 255] to [0.0, 1.0]
        img_array = img_array / 255.0  

        # 4. FIX: Z-Score Normalization for prediction consistency
        # Processed Pixel = (Pixel/255.0 - Mean) / StdDev
        img_array = (img_array - IMAGENET_MEANS) / IMAGENET_STDS
        
        # 5. Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    # FIX: Ensures the root URL works and does not return "Not Found"
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        # Returns the error shown in your previous images
        return render_template('index.html', prediction='Error: Model could not be loaded on the server.', error=True), 500
        
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # FIX: Generate a unique, safe filename to prevent link issues
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1]
        unique_filename = str(uuid.uuid4()) + file_extension
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        try:
            file.save(filepath)
        except Exception as e:
            print(f"Error saving file: {e}")
            return render_template('index.html', prediction='Error: Could not save image file.', error=True), 500

        processed_image = prepare_image(filepath)
        
        if processed_image is None:
            return render_template('index.html', prediction='Error: Image processing failed.', error=True), 400

        predictions = model.predict(processed_image)
        
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index] * 100
        predicted_label = CLASSES[predicted_index]

        result_text = (
            f"The predicted acne level is: **{predicted_label}**"
            f" (Confidence: {confidence:.2f}%)"
        )
        
        # Return the URL using the unique filename
        return render_template('index.html', 
                               prediction=result_text, 
                               image_url=url_for('static', filename=f'uploads/{unique_filename}'))

if __name__ == '__main__':
    # Add host='0.0.0.0' if running on a separate device, otherwise default is fine
    app.run(debug=True)