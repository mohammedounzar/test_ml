import numpy as np
import joblib
import tempfile
import os
from sklearn.ensemble import IsolationForest
import cloudinary
import cloudinary.uploader
import requests

# Function to train a model with received data
def train_user_model(user_id, records, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret, contamination=0.1):
    """
    Train an anomaly detection model for a specific user and upload it to Cloudinary.
    
    Parameters:
        user_id (str): The unique identifier for the user
        records (list): List of dictionaries containing the training data
                        Each record must have 'temperature', 'speed', and 'heart_beat' fields
        cloudinary_cloud_name (str): Cloudinary cloud name
        cloudinary_api_key (str): Cloudinary API key
        cloudinary_api_secret (str): Cloudinary API secret
        contamination (float): The proportion of outliers in the data set (default: 0.1)
        
    Returns:
        dict: Dictionary containing training result and model URL
    """
    try:
        # Validate records
        if len(records) < 10:
            return {
                "success": False,
                "error": "Insufficient data for training",
                "records_count": len(records),
                "min_required": 10
            }
        
        # Check that each record has the required fields
        for record in records:
            if not all(k in record for k in ['temperature', 'speed', 'heart_beat']):
                return {
                    "success": False,
                    "error": "Invalid record format",
                    "required_fields": ['temperature', 'speed', 'heart_beat']
                }
        
        # Extract features from records
        X = np.array([[r['temperature'], r['speed'], r['heart_beat']] for r in records])
        
        # Train the model
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X)
        
        # Configure Cloudinary
        cloudinary.config(
            cloud_name=cloudinary_cloud_name,
            api_key=cloudinary_api_key,
            api_secret=cloudinary_api_secret
        )
        
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            joblib.dump(model, temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                temp_file_path,
                resource_type="raw",
                public_id=f"models/{user_id}",
                overwrite=True
            )
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return {
                "success": True,
                "message": f"Model trained successfully for user {user_id}",
                "model_url": upload_result['secure_url'],
                "records_used": len(records)
            }
        
        except Exception as e:
            # Clean up temporary file in case of error
            os.unlink(temp_file_path)
            return {
                "success": False,
                "error": f"Error uploading model: {str(e)}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Error training model: {str(e)}"
        }

# Function to make predictions on records
def predict_anomalies(user_id, records, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret):
    """
    Make predictions on health monitoring data using a user's trained model.
    
    Parameters:
        user_id (str): The unique identifier for the user
        records (list): List of dictionaries containing the data to predict
                        Each record must have 'temperature', 'speed', and 'heart_beat' fields
        cloudinary_cloud_name (str): Cloudinary cloud name
        cloudinary_api_key (str): Cloudinary API key
        cloudinary_api_secret (str): Cloudinary API secret
        
    Returns:
        dict: Dictionary containing prediction results
              - success: Boolean indicating if prediction was successful
              - predictions: List of prediction results for each record
              - anomalies: List of records that were classified as anomalies
    """
    try:
        # Validate records
        if not records:
            return {
                "success": False,
                "error": "No data provided for prediction"
            }
        
        # Check that each record has the required fields
        for record in records:
            if not all(k in record for k in ['temperature', 'speed', 'heart_beat']):
                return {
                    "success": False,
                    "error": "Invalid record format",
                    "required_fields": ['temperature', 'speed', 'heart_beat']
                }
        
        # Download model from Cloudinary
        url = f"https://res.cloudinary.com/{cloudinary_cloud_name}/raw/upload/models/{user_id}.pkl"
        response = requests.get(url)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to download model for user {user_id}. Model may not exist or training may be required."
            }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        try:
            # Load model
            model = joblib.load(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            # Process each record
            predictions = []
            anomalies = []
            
            for record in records:
                # Extract features
                features = np.array([record['temperature'], record['speed'], record['heart_beat']]).reshape(1, -1)
                
                # Make prediction
                # Isolation Forest: 1 for normal, -1 for anomaly
                prediction = int(model.predict(features)[0])
                
                # Store result
                record_id = record.get('id', 'unknown')
                result = {
                    "id": record_id,
                    "is_anomaly": prediction == -1,
                    "prediction": prediction,
                    "timestamp": record.get('timestamp', None)
                }
                
                predictions.append(result)
                
                # If anomaly, add to anomalies list
                if prediction == -1:
                    anomalies.append({
                        "id": record_id,
                        "temperature": record['temperature'],
                        "speed": record['speed'],
                        "heart_beat": record['heart_beat'],
                        "timestamp": record.get('timestamp', None)
                    })
            
            return {
                "success": True,
                "predictions": predictions,
                "records_processed": len(records),
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies
            }
        
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
            return {
                "success": False,
                "error": f"Error making predictions: {str(e)}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Error making predictions: {str(e)}"
        }