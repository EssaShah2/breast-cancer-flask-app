import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Point template_folder to the parent directory's 'templates' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Safely load pickle models from root directory
model_path = os.path.join(BASE_DIR, 'diagonsis_detection.pkl')
encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')

pipeline = None
label_encoder = None

try:
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as err:
    print(f"Error loading pickle files: {err}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if pipeline is None or label_encoder is None:
            return jsonify({'error': 'Model files failed to load'}), 500

        data = request.get_json()
        features = data.get('features')
        
        if not features or len(features) != 30:
            return jsonify({'error': 'Expected exactly 30 numerical features'}), 400

        input_data = np.array(features).reshape(1, -1)
        raw_pred = pipeline.predict(input_data)
        probabilities = pipeline.predict_proba(input_data)[0]
        predicted_class = label_encoder.inverse_transform(raw_pred)[0]

        return jsonify({
            'status': 'success',
            'prediction': str(predicted_class),
            'confidence': float(np.max(probabilities))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Top-level export variable required by Vercel
app = app 