from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import traceback

# Import the separated model functions
from model_functions import train_user_model, predict_anomalies

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase
try:
    cred = credentials.Certificate("firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    # App already initialized
    pass

db = firestore.client()

# Get Cloudinary credentials from environment variables
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")


# first api to use
@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Endpoint to train a model for a specific user.
    
    Expected JSON format:
    {
        "user_id": "user123",
        "use_firebase": true,  // Optional, defaults to true
        "training_data": [     // Optional if use_firebase is true
            {
                "temperature": 36.8,
                "speed": 5.0,
                "heart_beat": 90
            },
            ...
        ]
    }
    
    Returns:
        JSON with training status
    """
    try:
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        if not data or 'user_id' not in data:
            return jsonify({"error": "Missing user_id"}), 400
        
        user_id = data['user_id']
        use_firebase = data.get('use_firebase', True)
        
        # Get training data
        if use_firebase:
            # Fetch data from Firebase
            collection = db.collection("users").document(user_id).collection("data").stream()
            records = []
            
            for doc in collection:
                record = doc.to_dict()
                if all(k in record for k in ['temperature', 'speed', 'heart_beat']):
                    records.append(record)
        else:
            # Use data provided in the request
            if 'training_data' not in data:
                return jsonify({"error": "Missing training_data"}), 400
            records = data['training_data']
        
        # Call the separated training function
        result = train_user_model(
            user_id=user_id,
            records=records,
            cloudinary_cloud_name=CLOUDINARY_CLOUD_NAME,
            cloudinary_api_key=CLOUDINARY_API_KEY,
            cloudinary_api_secret=CLOUDINARY_API_SECRET
        )
        
        if not result['success']:
            return jsonify({"success": False, "message": result.get('error', 'Training failed')}), 400
        
        # Record training info in Firebase
        db.collection("model_info").document(user_id).set({
            "trained_at": datetime.now().isoformat(),
            "records_count": result['records_used']
        })
        
        # Return simplified response
        return jsonify({
            "success": True,
            "message": "Model trained successfully"
        }), 200
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Error: " + str(e)
        }), 500

#  second api to use
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
        JSON with simplified prediction result (normal or not)
    """
    try:
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        if not data or 'user_id' not in data or 'data' not in data:
            return jsonify({"error": "Missing required fields: user_id, data"}), 400
        
        user_id = data['user_id']
        input_records = data['data']
        
        # Process records - add IDs and timestamps if missing
        records = []
        for record in input_records:
            record_copy = record.copy()
            
            # Generate record ID if not provided
            if 'id' not in record_copy:
                record_copy['id'] = str(uuid.uuid4())
            
            # Add timestamp if not provided
            if 'timestamp' not in record_copy:
                record_copy['timestamp'] = datetime.now().isoformat()
                
            records.append(record_copy)
        
        # Call the separated prediction function
        result = predict_anomalies(
            user_id=user_id,
            records=records,
            cloudinary_cloud_name=CLOUDINARY_CLOUD_NAME,
            cloudinary_api_key=CLOUDINARY_API_KEY,
            cloudinary_api_secret=CLOUDINARY_API_SECRET
        )
        
        if not result['success']:
            return jsonify({"success": False, "message": result.get('error', 'Prediction failed')}), 400
        
        # Store data and anomalies in Firebase - only if needed for your application
        for i, record in enumerate(records):
            record_id = record['id']
            
            # Store data in Firebase
            db.collection("users").document(user_id).collection("data").document(record_id).set(record)
            
            # If anomaly, store in anomalies collection
            prediction = result['predictions'][i]
            if prediction['is_anomaly']:
                anomaly_record = record.copy()
                anomaly_record['is_reviewed'] = False
                anomaly_record['data_id'] = record_id
                
                db.collection("users").document(user_id).collection("anomalies").document(record_id).set(anomaly_record)
        
        # Create simplified response
        simplified_results = []
        for prediction in result['predictions']:
            simplified_results.append({
                "is_normal": not prediction['is_anomaly']
            })
        
        # Return just one result if there's only one record
        if len(simplified_results) == 1:
            return jsonify({
                "success": True,
                "is_normal": simplified_results[0]['is_normal']
            }), 200
        else:
            return jsonify({
                "success": True,
                "results": simplified_results
            }), 200
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Error: " + str(e)
        }), 500

@app.route('/api/anomalies/<user_id>', methods=['GET'])
def get_anomalies(user_id):
    """
    Endpoint to retrieve anomalies for a specific user.
    """
    try:
        # Parse query parameters
        reviewed = request.args.get('reviewed')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Build query
        query = db.collection("users").document(user_id).collection("anomalies")
        
        if reviewed is not None:
            is_reviewed = reviewed.lower() == 'true'
            query = query.where("is_reviewed", "==", is_reviewed)
        
        # Order by timestamp (newest first)
        query = query.order_by("timestamp", direction=firestore.Query.DESCENDING)
        
        # Apply pagination
        query = query.limit(limit).offset(offset)
        
        # Execute query
        anomalies = []
        for doc in query.stream():
            anomaly = doc.to_dict()
            anomaly['id'] = doc.id
            anomalies.append(anomaly)
        
        # Return simplified response
        return jsonify({
            "success": True,
            "count": len(anomalies),
            "anomalies": anomalies
        }), 200
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Error: " + str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)