from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime, timedelta
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Load model and historical data
try:
    model = joblib.load('random_forest_model.pkl')
    historical_data = pd.read_csv('historical_passenger_data.csv')
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    logging.info("Model and historical data loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model or data: {str(e)}")
    raise

# Helper function to parse user input
def parse_user_input(text):
    text = text.lower()
    features = {
        'day': None,
        'weather': None,
        'hour': None,
        'peak_hours': 'No',
        'weekends': 'No',
        'holidays': 'No',
        'date': None
    }
    
    # Detect date
    if 'yesterday' in text:
        features['date'] = (datetime.now() - timedelta(days=1)).date()
    elif 'tomorrow' in text:
        features['date'] = (datetime.now() + timedelta(days=1)).date()
    else:
        date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
        if date_match:
            try:
                features['date'] = pd.to_datetime(date_match.group(1)).date()
            except ValueError:
                return features, "Invalid date format"
    
    # Detect weather
    weather_map = {'sunny': 'Sunny', 'rainy': 'Rainy', 'clear': 'Clear', 'cloudy': 'Cloudy'}
    for key, value in weather_map.items():
        if key in text:
            features['weather'] = value
            break
    
    # Detect hour
    hour_match = re.search(r'\b(\d{1,2})(?:\s*(?:am|pm))?\b', text)
    if hour_match:
        try:
            hour = int(hour_match.group(1))
            if 'pm' in text and hour != 12:
                hour += 12
            if 'am' in text and hour == 12:
                hour = 0
            features['hour'] = hour
        except ValueError:
            return features, "Invalid hour format"
    
    # Detect holiday
    if 'holiday' in text:
        features['holidays'] = 'Yes'
    
    # Fill date-related features
    if features['date']:
        try:
            dt = pd.to_datetime(features['date'])
            features['day'] = dt.strftime('%A')
            features['weekends'] = 'Yes' if dt.weekday() >= 5 else 'No'
            if features['date'] == pd.to_datetime('2024-10-14').date():
                features['holidays'] = 'Yes'
        except Exception as e:
            return features, f"Date processing error: {str(e)}"
    
    # Detect peak hours
    if features['hour'] is not None:
        if 7 <= features['hour'] <= 9 or 16 <= features['hour'] <= 18:
            features['peak_hours'] = 'Yes'
    
    return features, None

# Helper function to query historical data
def get_historical_passengers(features):
    if not features['date']:
        return None, "Please specify a date"
    query_date = pd.to_datetime(features['date'])
    query = historical_data[historical_data['date'].dt.date == query_date]
    
    if features['hour'] is not None:
        query = query[query['time_value'].str.startswith(str(features['hour']).zfill(2))]
    
    if features['weather']:
        query = query[query['weather'] == features['weather']]
    
    if features['holidays'] == 'Yes':
        query = query[query['holidays'] == 'Yes']
    
    if query.empty:
        return None, "No historical data found for the specified conditions"
    
    avg_passengers = query['passengers'].mean()
    return avg_passengers, f"Historical average: {avg_passengers:.0f} passengers"

# Helper function to predict passengers
def predict_passengers(features):
    required_features = ['day', 'weather', 'hour', 'peak_hours', 'weekends', 'holidays']
    for f in required_features:
        if features[f] is None:
            return None, f"Missing feature: {f}"
    
    try:
        # Create DataFrame with raw categorical values
        input_df = pd.DataFrame([[
            features['day'],
            features['weather'],
            features['peak_hours'],
            features['weekends'],
            features['holidays'],
            features['hour']
        ]], columns=['day', 'weather', 'peak_hours', 'weekends', 'holidays', 'hour'])
        
        # Use pipeline's predict method (handles encoding)
        prediction = model.predict(input_df)[0]
        message = f"Predicted passengers: {prediction:.0f}"
        if prediction > 50:
            message += ". Warning: High passenger count"
        return prediction, message
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# Test route
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running"})

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received request data: {data}")
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'text' in data:
            features, error = parse_user_input(data['text'])
            if error:
                return jsonify({'error': error}), 400
            
            if features['date'] and features['date'] <= datetime.now().date():
                result, message = get_historical_passengers(features)
            else:
                result, message = predict_passengers(features)
        else:
            return jsonify({'error': 'Text-based input required'}), 400
        
        if result is None:
            return jsonify({'error': message}), 400
        
        return jsonify({'predicted_passengers': int(result), 'message': message})
    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)