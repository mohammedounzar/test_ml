from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import traceback

# Import the updated prediction functions for LSTM-OC-SVM
from predict import predict_anomalies

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase
try:
    cred = credentials.Certificate("firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    pass

db = firestore.client()

# Get Cloudinary credentials from environment variables
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Prediction API for LSTM-OC-SVM model
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions using the LSTM-OC-SVM model.
    
    Expected JSON format:
    {
        "user_id": "user123",
        "data": [
            {
                "temperature": 36.8,
                "speed": 5.0,
                "heart_beat": 90,
                "timestamp": "2024-01-01T10:00:00" (optional)
            }
        ],
        "sequence_length": 10 (optional, default: 10)
    }
    
    Returns:
        JSON with prediction result: success, user_id, results
    """
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data or 'data' not in data:
            return jsonify({"error": "Missing required fields: user_id, data"}), 400
        
        user_id = data['user_id']
        input_records = data['data']
        sequence_length = data.get('sequence_length', 10)  # Default sequence length
        
        # Validate sequence length
        if sequence_length < 1 or sequence_length > 100:
            return jsonify({"error": "sequence_length must be between 1 and 100"}), 400
        
        # Process records and add missing fields
        records = []
        for i, record in enumerate(input_records):
            record_copy = record.copy()
            if 'id' not in record_copy:
                record_copy['id'] = str(uuid.uuid4())
            if 'timestamp' not in record_copy:
                # Use sequential timestamps if not provided
                base_time = datetime.now()
                record_copy['timestamp'] = base_time.replace(second=i).isoformat()
            records.append(record_copy)
        
        # Use LSTM-OC-SVM prediction with sequence processing
        result = predict_anomalies(
            user_id=user_id,
            records=records,
            cloudinary_cloud_name=CLOUDINARY_CLOUD_NAME,
            cloudinary_api_key=CLOUDINARY_API_KEY,
            cloudinary_api_secret=CLOUDINARY_API_SECRET,
            sequence_length=sequence_length
        )

        print(f"Prediction result for user {user_id}: {result}")
        
        if not result['success']:
            return jsonify({
                "success": False, 
                "user_id": user_id,
                "message": result.get('error', 'Prediction failed')
            }), 400
        
        # Simplified results matching requested format
        simplified_results = []
        for prediction in result['predictions']:
            simplified_results.append({
                "id": prediction.get('id'),
                "is_anomaly": prediction['is_anomaly']
            })
        
        # Return format based on number of results
        if len(simplified_results) == 1:
            return jsonify({
                "success": True,
                "user_id": user_id,
                "is_anomaly": simplified_results[0]['is_anomaly']
            }), 200
        else:
            return jsonify({
                "success": True,
                "user_id": user_id,
                "results": simplified_results
            }), 200
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Error: " + str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)