import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pcb_defect_model.h5')
model = keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)[0][0]
        is_defect = bool(pred > 0.5)
        confidence = float(pred) if is_defect else float(1 - pred)
        return jsonify({
            'result': 'DEFECT DETECTED' if is_defect else 'NO DEFECT',
            'is_defect': is_defect,
            'confidence': round(confidence * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
