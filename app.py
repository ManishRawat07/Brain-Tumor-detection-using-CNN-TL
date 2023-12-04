from flask import Flask, render_template, request, url_for
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import cv2
import os
import tensorflow as tf
import cv2

BUCKET_NAME = "btumordetect1"

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('effnet.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the 'image' file is present in the request.files
    if 'image' in request.files:
        # Get the image file
        img_file = request.files['image']

        # Read image using PIL
        img = Image.open(img_file)

        # Save the image temporarily
        temp_image_path = 'temp_upload.png'
        img.save(temp_image_path)

        # Preprocess the image as needed
        new_image = cv2.resize(np.array(img), (150, 150))
        # new_image = new_image / 255.0  # Normalize the image

        # Make predictions
        new_image = np.expand_dims(new_image, axis=0)

        # Make a prediction
        predictions = model.predict(new_image)
        predicted_class_index = np.argmax(predictions)
        # print(predictions)
        # Map the predicted class index to the corresponding label
        labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
        predicted_tumor_type = labels[predicted_class_index]

        # Process the prediction and return the result
        # ...

        # Pass the image URL to the result template
        image_url = temp_image_path

        # Remove the temporarily saved image
        os.remove(temp_image_path)

        return render_template('result.html', result=predicted_tumor_type, image_url=image_url)
    else:
        return "No file uploaded."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
