from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import traceback

# Import the separated model function
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

# Only API kept: Prediction API
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions for a specific user.
    
    Expected JSON format:
    {
        "user_id": "user123",
        "data": [
            {
                "temperature": 36.8,
                "speed": 5.0,
                "heart_beat": 90
            }
        ]
    }
    
    Returns:
        JSON with prediction result including user_id
    """
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data or 'data' not in data:
            return jsonify({"error": "Missing required fields: user_id, data"}), 400
        
        user_id = data['user_id']
        input_records = data['data']
        
        records = []
        for record in input_records:
            record_copy = record.copy()
            if 'id' not in record_copy:
                record_copy['id'] = str(uuid.uuid4())
            if 'timestamp' not in record_copy:
                record_copy['timestamp'] = datetime.now().isoformat()
            records.append(record_copy)
        
        result = predict_anomalies(
            user_id=user_id,
            records=records,
            cloudinary_cloud_name=CLOUDINARY_CLOUD_NAME,
            cloudinary_api_key=CLOUDINARY_API_KEY,
            cloudinary_api_secret=CLOUDINARY_API_SECRET
        )
        
        if not result['success']:
            return jsonify({
                "success": False, 
                "user_id": user_id,
                "message": result.get('error', 'Prediction failed')
            }), 400
        
        simplified_results = []
        for prediction in result['predictions']:
            simplified_results.append({
                "is_normal": not prediction['is_anomaly']
            })
        
        if len(simplified_results) == 1:
            return jsonify({
                "success": True,
                "user_id": user_id,
                "is_normal": simplified_results[0]['is_normal']
            }), 200
        else:
            return jsonify({
                "success": True,
                "user_id": user_id,
                "results": simplified_results
            }), 200
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Error: " + str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)