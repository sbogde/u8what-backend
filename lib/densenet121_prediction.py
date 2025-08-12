import numpy as np
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = DenseNet121()  
model.summary()

def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_image(img_path):
    img = load_and_preprocess_image(img_path)
    preds = model.predict(img)
    return decode_predictions(preds, top=5)[0]
