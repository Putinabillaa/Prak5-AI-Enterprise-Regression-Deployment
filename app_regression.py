from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os

# Flask app
app = Flask(__name__)
CORS(app)

# Prediction endpoint
@app.route('/predict_regression', methods=['POST'])
def predict():
    data = request.json
    year = data['year']
    proglang = data['programmingLanguage']

    # Load the serialized model based on the programming language
    model_path = os.path.join('models', proglang, f'{proglang}_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model not found for programming language {proglang}'}), 404

    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    model = model_info['model']
    scaler = model_info['scaler']

    # Perform the same preprocessing on input data
    year_scaled = scaler.transform([[year]])
    # Make prediction
    prediction = model.predict(year_scaled)[0]
    return jsonify({'prediction': prediction})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)