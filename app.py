from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the saved model
model = tf.keras.models.load_model('//image_classification/model.h5')

# Allow only certain file extensions (like images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the input image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0
    return image

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    return open_cv_image[:, :, ::-1]  # Convert RGB to BGR (OpenCV uses BGR)

# Calculate defect percentage
def calculate_defect_percentage(cv2_image):
    grayscale_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    _, defect_mask = cv2.threshold(grayscale_img, 100, 255, cv2.THRESH_BINARY_INV)
    
    total_pixels = grayscale_img.size
    defective_pixels = np.sum(defect_mask == 255)
    
    percentage_defect = (defective_pixels / total_pixels) * 100
    return percentage_defect

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or len(request.files.getlist('file')) == 0:
        return jsonify({'error': 'No file uploaded or file format not supported'}), 400

    files = request.files.getlist('file')
    results = []

    for file in files:
        if file and allowed_file(file.filename):
            # Open the image and process in-memory without saving
            pil_image = Image.open(file.stream)
            cv2_image = pil_to_cv2(pil_image)

            # Calculate defect percentage
            defect_percent = float(calculate_defect_percentage(cv2_image))
            defect_percent = float(defect_percent)

            # Preprocess image for model
            processed_image = preprocess_image(pil_image, target_size=(256, 256))

            # Make predictions
            predictions = model.predict(processed_image)
            predicted_class = "defect" if predictions <= 0.5 else "non-defect"

             # Convert image to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            #if product is defect and defect_percent is < 20% then it is less defect and else high defect
            #if product is non -defect then status is good.

            if predicted_class == 'defect' and defect_percent < 20: 
                    status = "Defect is fixable. Please contact Customer Support."
            elif predicted_class == 'defect' and defect_percent > 20: 
                    status = "Defect is of great concern. Please contact Manufacturing support unit."
            else: 
                 status = "Your product looks good."

            results.append({
               'image_url': f'data:image/jpeg;base64,{encoded_image}',  # Base64 encoded image to display in the frontend
                'predicted_class': predicted_class,
                'status': status
                
            })

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
