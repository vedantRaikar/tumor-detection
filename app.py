from flask import Flask, render_template, request
import tensorflow as tf 
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.utils import load_img, img_to_array

app = Flask(__name__)
model = load_model(r'C:\Users\vedant raikar\Desktop\tumor-detection\model\model_weights.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    # Specify the path to save the uploaded image
    img_path = r'C:\Users\vedant raikar\Desktop\tumor-detection\static\test_img.jpg'
    img.save(img_path)

    img = load_img(img_path, target_size=(64, 64))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    prediction = model.predict(x)
    predicted_class = np.argmax(prediction[0])
    class_labels = ['No Tumor', 'Tumor']
    predicted_label = class_labels[predicted_class]

    return render_template('index.html', prediction=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)
