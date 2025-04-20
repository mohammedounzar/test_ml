from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import uuid
import joblib
import requests
import tempfile
import os
from datetime import datetime
from anomaly_utils import train_anomaly_model

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase-adminsdk.json")
try:
    firebase_admin.initialize_app(cred)
except ValueError:
    # App already initialized
    pass
db = firestore.client()

@app.route('/api/data', methods=['POST'])
def receive_data():
    """
    Endpoint to receive user data and store it in Firestore.
    Also checks if the data is anomalous and stores the prediction.
    
    Expected JSON format:
    {
        "user_id": "user123",
        "temperature": 36.8,
        "speed": 5.0,
        "heart_beat": 90,
        "altitude": 100,
        "longitude": 12.345,
        "latitude": 45.678
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data:
            return jsonify({"error": "Invalid data format or missing user_id"}), 400
        
        # Extract required fields
        user_id = data['user_id']
        
        # Validate required fields
        required_fields = ['temperature', 'speed', 'heart_beat']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Store data in Firestore
        db.collection("users").document(user_id).collection("data").document(doc_id).set(data)
        
        # Predict anomaly
        prediction = predict_anomaly(user_id, data)
        
        # If anomalous, store in anomalies collection
        if prediction == -1:  # -1 is anomaly in Isolation Forest
            anomaly_data = {
                "data_id": doc_id,
                "timestamp": data['timestamp'],
                "is_reviewed": False,
                "temperature": data['temperature'],
                "speed": data['speed'],
                "heart_beat": data['heart_beat']
            }
            
            # Add optional fields if they exist
            for field in ['altitude', 'longitude', 'latitude']:
                if field in data:
                    anomaly_data[field] = data[field]
            
            db.collection("users").document(user_id).collection("anomalies").document(doc_id).set(anomaly_data)
            
            return jsonify({
                "success": True,
                "message": "Data received and stored",
                "anomaly_detected": True,
                "data_id": doc_id
            }), 201
        
        return jsonify({
            "success": True,
            "message": "Data received and stored",
            "anomaly_detected": False,
            "data_id": doc_id
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/anomalies/<user_id>', methods=['GET'])
def get_anomalies(user_id):
    """
    Endpoint to retrieve anomalies for a specific user.
    Optional query parameter 'reviewed' to filter by review status.
    """
    try:
        reviewed = request.args.get('reviewed', None)
        
        query = db.collection("users").document(user_id).collection("anomalies")
        
        if reviewed is not None:
            is_reviewed = reviewed.lower() == 'true'
            query = query.where("is_reviewed", "==", is_reviewed)
        
        # Order by timestamp (newest first)
        query = query.order_by("timestamp", direction=firestore.Query.DESCENDING)
        
        anomalies = []
        for doc in query.stream():
            anomaly = doc.to_dict()
            anomaly['id'] = doc.id
            anomalies.append(anomaly)
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "anomalies": anomalies,
            "count": len(anomalies)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/anomalies/<user_id>/<anomaly_id>', methods=['GET'])
def get_anomaly_details(user_id, anomaly_id):
    """
    Endpoint to retrieve details of a specific anomaly.
    """
    try:
        doc_ref = db.collection("users").document(user_id).collection("anomalies").document(anomaly_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Anomaly not found"}), 404
        
        anomaly = doc.to_dict()
        anomaly['id'] = doc.id
        
        return jsonify({
            "success": True,
            "anomaly": anomaly
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/anomalies/<user_id>/<anomaly_id>', methods=['PATCH'])
def update_anomaly_status(user_id, anomaly_id):
    """
    Endpoint to update the review status of an anomaly.
    
    Expected JSON format:
    {
        "is_reviewed": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'is_reviewed' not in data:
            return jsonify({"error": "Missing is_reviewed field"}), 400
        
        doc_ref = db.collection("users").document(user_id).collection("anomalies").document(anomaly_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Anomaly not found"}), 404
        
        # Update only the is_reviewed field
        doc_ref.update({"is_reviewed": data['is_reviewed']})
        
        return jsonify({
            "success": True,
            "message": "Anomaly review status updated"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def predict_anomaly(user_id, data):
    """
    Predict if the data is anomalous using the user's model.
    If no model exists, train a new one using existing data.
    
    Returns:
        1 for normal data
        -1 for anomalous data
    """
    try:
        # Try to download the user's model from Cloudinary
        features = [data['temperature'], data['speed'], data['heart_beat']]
        
        try:
            # Try to use existing model
            url = f"https://res.cloudinary.com/ddnkpgyqv/raw/upload/models/{user_id}.pkl"
            response = requests.get(url)
            
            if response.status_code == 200:
                # Save the model temporarily
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                # Load the model and make prediction
                model = joblib.load(temp_file_path)
                os.unlink(temp_file_path)  # Delete temp file
                
                return model.predict(np.array(features).reshape(1, -1))[0]
            
        except Exception as e:
            print(f"Error downloading or using model: {e}")
            # If model doesn't exist or error occurred, train a new one
            pass
        
        # If we reached here, we need to train a new model
        return train_and_predict(user_id, features)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Default to normal if something goes wrong
        return 1

def train_and_predict(user_id, features):
    """
    Train a new model and make a prediction.
    """
    # Fetch all user data
    collection = db.collection("users").document(user_id).collection("data").stream()
    records = []
    
    for doc in collection:
        data = doc.to_dict()
        if all(k in data for k in ['temperature', 'speed', 'heart_beat']):
            records.append(data)
    
    if len(records) < 3:
        # Not enough data to train, assume normal
        return 1
    
    # Train model
    model = train_anomaly_model(records)
    
    # Save model temporarily
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
        joblib.dump(model, temp_file.name)
        temp_file_path = temp_file.name
    
    # Make prediction
    prediction = model.predict(np.array(features).reshape(1, -1))[0]
    
    # Upload model to Cloudinary (if credentials are available)
    try:
        import cloudinary
        import cloudinary.uploader
        
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET")
        )
        
        cloudinary.uploader.upload(
            temp_file_path,
            resource_type="raw",
            public_id=f"models/{user_id}",
            overwrite=True
        )
    except Exception as e:
        print(f"Error uploading model to Cloudinary: {e}")
    
    # Delete temp file
    os.unlink(temp_file_path)
    
    return prediction

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)