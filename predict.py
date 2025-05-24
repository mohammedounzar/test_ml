"""
Prediction script for LSTM-OC-SVM anomaly detection
Compatible with the joint training implementation from the paper
"""

import numpy as np
import torch
import joblib
import tempfile
import os
import requests
import cloudinary
from lstm_ocsvm_trainer import LSTMOCSVMJoint, VariableLengthDataset, collate_variable_length
from torch.utils.data import DataLoader


def _configure_cloudinary(cloud_name, api_key, api_secret):
    """Configure Cloudinary once"""
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret
    )


def records_to_sequences(records, sequence_length=10):
    """
    Convert individual records to sequences for LSTM processing
    Optimized for real-time mobile data collection patterns
    
    Args:
        records: List of dicts with 'temperature', 'speed', 'heart_beat'
        sequence_length: Length of sequences to create for LSTM
    
    Returns:
        sequences: List of numpy arrays (sequences)
        record_indices: List mapping each sequence to its source records
    """
    if len(records) < sequence_length:
        # If we have fewer records than sequence_length, use all records as one sequence
        # This is common for real-time mobile data collection
        features = []
        for r in records:
            features.append([r["temperature"], r["speed"], r["heart_beat"]])
        return [np.array(features)], [list(range(len(records)))]
    
    sequences = []
    record_indices = []
    
    # For mobile real-time data, use a sliding window approach
    # This provides better anomaly detection for streaming data
    step_size = max(1, sequence_length // 4)  # 75% overlap for better coverage
    
    for i in range(0, len(records) - sequence_length + 1, step_size):
        features = []
        indices = []
        for j in range(sequence_length):
            r = records[i + j]
            features.append([r["temperature"], r["speed"], r["heart_beat"]])
            indices.append(i + j)
        
        sequences.append(np.array(features))
        record_indices.append(indices)
    
    # If we have remaining records, create one more sequence
    if len(records) > sequence_length:
        remaining_start = len(records) - sequence_length
        if remaining_start not in [idx[0] for idx in record_indices]:
            features = []
            indices = []
            for j in range(sequence_length):
                r = records[remaining_start + j]
                features.append([r["temperature"], r["speed"], r["heart_beat"]])
                indices.append(remaining_start + j)
            
            sequences.append(np.array(features))
            record_indices.append(indices)
    
    return sequences, record_indices


def predict_anomalies(
    user_id,
    records,
    cloudinary_cloud_name,
    cloudinary_api_key,
    cloudinary_api_secret,
    public_id="models/lstm_ocsvm_joint",
    sequence_length=10
):
    """
    Download the LSTM-OC-SVM model from Cloudinary and predict anomalies.

    Parameters:
      - user_id: str, identifier for the user
      - records: list of dicts, each with 'temperature','speed','heart_beat'
      - cloudinary_cloud_name: str, Cloudinary cloud name
      - cloudinary_api_key: str, Cloudinary API key
      - cloudinary_api_secret: str, Cloudinary API secret
      - public_id: same ID used at training time (default: "models/lstm_ocsvm_joint")
      - sequence_length: int, length of sequences to create for LSTM processing

    Returns:
      dict with:
        - success: bool
        - user_id: str
        - predictions: list of {id, is_anomaly, prediction, timestamp, anomaly_score}
        - anomalies: list of record dicts flagged as anomalies
        - error: str (on failure)
    """
    if not records:
        return {"success": False, "user_id": user_id, "error": "No records provided"}

    # Validate record format
    for r in records:
        if not all(k in r for k in ("temperature", "speed", "heart_beat")):
            return {"success": False, "user_id": user_id, "error": "Invalid record format"}

    # Download model file from Cloudinary
    print(f"Downloading model from Cloudinary: {public_id}")
    url = f"https://res.cloudinary.com/{cloudinary_cloud_name}/raw/upload/{public_id}.pkl"
    
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            return {
                "success": False, 
                "user_id": user_id, 
                "error": f"Could not fetch model (status {resp.status_code}); re-train required"
            }
    except Exception as e:
        return {
            "success": False, 
            "user_id": user_id, 
            "error": f"Network error downloading model: {str(e)}"
        }

    # Write to temp file and load model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        # Load the LSTM-OC-SVM model
        model = joblib.load(tmp_path)
        
        # Verify it's the correct model type
        if not isinstance(model, LSTMOCSVMJoint):
            return {
                "success": False, 
                "user_id": user_id, 
                "error": f"Invalid model type: expected LSTMOCSVMJoint, got {type(model)}"
            }
        
        print(f"Loaded LSTM-OC-SVM model successfully")
        
    except Exception as e:
        return {
            "success": False, 
            "user_id": user_id, 
            "error": f"Failed to load model: {str(e)}"
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    try:
        # Convert records to sequences for LSTM processing
        sequences, record_indices = records_to_sequences(records, sequence_length)
        print(f"Created {len(sequences)} sequences from {len(records)} records")
        
        # Create dataset and dataloader
        dataset = VariableLengthDataset(sequences)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, 
                              collate_fn=collate_variable_length)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get predictions and scores
        all_predictions = []
        all_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Use the model's predict method (implements Equation 10)
                batch_predictions, batch_scores = model.predict(batch)
                all_predictions.extend(batch_predictions.cpu().numpy())
                all_scores.extend(batch_scores.cpu().numpy())
        
        print(f"Generated {len(all_predictions)} predictions")
        
    except Exception as e:
        return {
            "success": False, 
            "user_id": user_id, 
            "error": f"Prediction failed: {str(e)}"
        }

    # Process predictions and create response
    predictions = []
    anomalies = []
    
    # Create a mapping from record index to anomaly info
    record_anomaly_info = {}
    
    for seq_idx, (pred, score) in enumerate(zip(all_predictions, all_scores)):
        # Each sequence prediction affects multiple records
        affected_indices = record_indices[seq_idx]
        
        for record_idx in affected_indices:
            if record_idx not in record_anomaly_info:
                record_anomaly_info[record_idx] = []
            
            record_anomaly_info[record_idx].append({
                'prediction': int(pred),
                'score': float(score),
                'is_anomaly': (pred == -1)
            })
    
    # Aggregate predictions for each record
    for i, record in enumerate(records):
        if i in record_anomaly_info:
            # If multiple sequences include this record, use weighted average
            seq_predictions = record_anomaly_info[i]
            
            # For mobile real-time data, use weighted scoring
            # Give more weight to recent predictions (higher sequence indices)
            weights = [1.0 + 0.1 * j for j in range(len(seq_predictions))]
            total_weight = sum(weights)
            
            # Weighted average of scores
            weighted_score = sum(p['score'] * w for p, w in zip(seq_predictions, weights)) / total_weight
            
            # Anomaly decision based on weighted score and majority vote
            anomaly_votes = sum(w for p, w in zip(seq_predictions, weights) if p['is_anomaly'])
            is_anomaly = (anomaly_votes > total_weight / 2) or (weighted_score < -0.5)  # Threshold for anomaly
            
            final_prediction = -1 if is_anomaly else 1
            
        else:
            # Record not included in any sequence (shouldn't happen with current logic)
            is_anomaly = False
            weighted_score = 0.0
            final_prediction = 1
        
        pred_record = {
            "id": record.get("id"),
            "is_anomaly": is_anomaly,
            "prediction": final_prediction,
            "anomaly_score": weighted_score,
            "timestamp": record.get("timestamp")
        }
        
        predictions.append(pred_record)
        
        if is_anomaly:
            anomalies.append(record)

    print(f"Found {len(anomalies)} anomalies out of {len(records)} records")

    return {
        "success": True,
        "user_id": user_id,
        "predictions": predictions,
        "anomalies": anomalies,
        "model_info": {
            "type": "LSTM-OC-SVM",
            "sequence_length": sequence_length,
            "total_sequences": len(sequences),
            "overlap_strategy": "sliding_window"
        }
    }


def predict_single_sequence(
    user_id,
    sequence_data,
    cloudinary_cloud_name,
    cloudinary_api_key,
    cloudinary_api_secret,
    public_id="models/lstm_ocsvm_joint"
):
    """
    Predict anomaly for a single sequence (alternative interface)
    
    Args:
        user_id: User identifier
        sequence_data: List of lists [[temp, speed, heart], [temp, speed, heart], ...]
        Other args: Same as predict_anomalies
    
    Returns:
        dict with prediction result for the sequence
    """
    if not sequence_data:
        return {"success": False, "user_id": user_id, "error": "No sequence data provided"}

    # Download and load model (same as above)
    url = f"https://res.cloudinary.com/{cloudinary_cloud_name}/raw/upload/{public_id}.pkl"
    
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            return {"success": False, "user_id": user_id, "error": "Could not fetch model"}
    except Exception as e:
        return {"success": False, "user_id": user_id, "error": f"Network error: {str(e)}"}

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        model = joblib.load(tmp_path)
        model.eval()
        
        # Convert to tensor and predict
        sequence_tensor = torch.tensor([sequence_data], dtype=torch.float32)
        
        with torch.no_grad():
            prediction, score = model.predict(sequence_tensor)
            
        return {
            "success": True,
            "user_id": user_id,
            "is_anomaly": (prediction[0] == -1),
            "prediction": int(prediction[0]),
            "anomaly_score": float(score[0])
        }
        
    except Exception as e:
        return {"success": False, "user_id": user_id, "error": f"Prediction failed: {str(e)}"}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)