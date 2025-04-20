from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    pipeline = joblib.load('random_forest_model.pkl')
    logging.info("Model pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logging.info(f"Received data: {data}")
        
        # Validate required features
        required_features = ['day', 'weather', 'peak_hours', 'weekends', 'holidays', 'hour']
        for feature in required_features:
            if feature not in data:
                raise ValueError(f"Missing feature: {feature}")
            if not isinstance(data[feature], (int, float)):
                raise ValueError(f"Feature {feature} must be a number, got {type(data[feature])}")
        
        # Create DataFrame with explicit types matching training
        features = pd.DataFrame([data], columns=required_features)
        # Convert categorical features to strings (as OneHotEncoder may expect strings)
        categorical_features = ['day', 'weather', 'peak_hours', 'weekends', 'holidays']
        for col in categorical_features:
            features[col] = features[col].astype(str)
        features['hour'] = features['hour'].astype(int)
        
        logging.info(f"Features DataFrame: {features.to_dict()}")
        logging.info(f"Feature dtypes: {features.dtypes.to_dict()}")
        
        # Predict
        prediction = pipeline.predict(features)[0]
        logging.info(f"Prediction: {prediction}")
        return jsonify({'predicted_passengers': int(prediction)})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
