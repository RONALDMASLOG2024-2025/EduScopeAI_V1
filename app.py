import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

app = Flask(__name__)

# Load model and class labels
model = load_model('model/microorganism_model.keras')
with open("model/class_names.json", "r") as f:
    classes = json.load(f)

img_size = (224, 224)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            path = os.path.join('static', 'upload.jpg')
            image.save(path)

            img = load_img(path, target_size=img_size)
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0]
            label_index = np.argmax(prediction)
            
            if label_index < len(classes):
                label = classes[label_index]
                confidence = round(100 * np.max(prediction), 2)
                result = f"{label} ({confidence}%)"
            else:
                result = "Unknown organism detected"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
