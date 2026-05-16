from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pcb_defect_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Predict
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
