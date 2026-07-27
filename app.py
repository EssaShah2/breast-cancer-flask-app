from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# --- ADD THE TWO LINES HERE ---
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# ------------------------------

# Load saved pickle artifacts
with open('diagonsis_detection.pkl', 'rb') as f:
    pipeline = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)