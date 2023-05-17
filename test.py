import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img , img_to_array
# Specify the path to the saved model
model_path = r'C:\Users\vedant raikar\Desktop\tumor-detection\model_weights.h5'

# Specify the path to the test image
test_image_path = r'C:\Users\vedant raikar\Desktop\tumor-detection\test3.jpg'

# Load the saved model
model = load_model(model_path)

# Load and preprocess the test image
img = load_img(test_image_path, target_size=(64, 64))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Make predictions on the test image
predictions = model.predict(x)

# Convert the prediction probabilities to class labels
class_labels = ['NO Tumor', 'Tumor']
predicted_class = np.argmax(predictions[0])
predicted_label = class_labels[predicted_class]

# Print the predicted class label
print('Predicted Class:', predicted_label)
