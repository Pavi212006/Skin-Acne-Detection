**Skin Acne Detection System**
This is a web-based application that uses a Convolutional Neural Network (CNN) to detect and classify skin acne from images. The project is built using Python, Flask, and TensorFlow/Keras.

**Features**
 * Image Upload: Users can upload images of skin affected by acne.
 * Deep Learning Prediction: A trained CNN model analyzes the image and provides results.
 * Web Interface: A user-friendly interface built with HTML/CSS.

**Tech Stack**
 * Backend: Flask (Python)
 * Deep Learning: TensorFlow, Keras
 * Frontend: HTML5, CSS3
 * Environment: Virtual Environment (venv)

**Installation & Setup**
 * Clone the repository:
   git clone https://github.com/Pavi212006/Skin-Acne-Detection.git
cd Skin-Acne-Detection

 * Install Dependencies:
   Make sure you have Python installed, then run:
   pip install -r requirements.txt

 * Download the Model File:
   Due to GitHub's file size limits, the trained model (.h5) is hosted on Google Drive.
   * DOWNLOAD THE MODEL HERE: https://drive.google.com/file/d/19H6aYr4m2POV7okmZzXwi4JOJQ7Dv5Pn/view?usp=sharing
   * After downloading, place the acne_cnn_model.h5 file in the main project folder.
 * Run the Application:
   python app.py

   Open your browser and go to http://127.0.0.1:5000.

**Project Structure**
 * app.py: The main Flask application script.
 * static/: Contains uploaded images and CSS files.
 * templates/: Contains the index.html for the web interface.
 * acne_cnn_model.h5: The trained deep learning model (to be downloaded separately).
 * requirements.txt: List of necessary Python libraries.
