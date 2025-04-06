import os
import gdown  # type: ignore
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Correct target path where model will be saved
MODEL_DIR = "model"
MODEL_PATH = "model/eye_disease_final.h5"
GDRIVE_FILE_ID = "14naPo5TfH9dboUPeWcsdyUSp7lNcH3me"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Make sure the directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)


from flask import Flask, render_template, request
import os
import uuid
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Hypertension',
               'Macular Degeneration', 'Myopia', 'Normal', 'Others']

model = load_model('model/eye_disease_final.h5')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_files = request.files.getlist("image")
    results = []


    for file in uploaded_files:
        if file.filename != '':
            unique_filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)[0]
            result = {class_names[i]: f"{predictions[i]*100:.2f}%" for i in range(len(class_names))}
            results.append({"filename": file.filename, "result": result})

    return render_template('index.html', results=results)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
