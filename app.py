from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Load the model
try:
    model = joblib.load('random_forest_model.pkl')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received request: {data}")
        # Convert input to DataFrame with expected feature order
        features = pd.DataFrame([data], columns=['day', 'weather', 'peak_hours', 'weekends', 'holidays', 'hour'])
        prediction = model.predict(features)[0]
        logging.info(f"Prediction: {prediction}")
        return jsonify({'predicted_passengers': int(prediction)})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)