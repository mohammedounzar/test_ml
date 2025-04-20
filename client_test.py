import requests
import json
import argparse
import random
import datetime

# API base URL
BASE_URL = "http://localhost:5000/api"

def train_model(user_id, use_firebase=True, sample_data=None):
    """Train a model for a specific user"""
    url = f"{BASE_URL}/train"
    
    payload = {
        "user_id": user_id,
        "use_firebase": use_firebase
    }
    
    if not use_firebase and sample_data:
        payload["training_data"] = sample_data
    
    response = requests.post(url, json=payload)
    return response.json()

def make_predictions(user_id, data):
    """Make predictions for a set of data records"""
    url = f"{BASE_URL}/predict"
    
    payload = {
        "user_id": user_id,
        "data": data
    }
    
    response = requests.post(url, json=payload)
    return response.json()

def get_anomalies(user_id, reviewed=None, limit=100, offset=0):
    """Get anomalies for a specific user"""
    url = f"{BASE_URL}/anomalies/{user_id}"
    
    params = {
        "limit": limit,
        "offset": offset
    }
    
    if reviewed is not None:
        params["reviewed"] = "true" if reviewed else "false"
    
    response = requests.get(url, params=params)
    return response.json()

def get_anomaly_details(user_id, anomaly_id):
    """Get details for a specific anomaly"""
    url = f"{BASE_URL}/anomalies/{user_id}/{anomaly_id}"
    
    response = requests.get(url)
    return response.json()

def update_anomaly_status(user_id, anomaly_id, is_reviewed):
    """Update the review status of an anomaly"""
    url = f"{BASE_URL}/anomalies/{user_id}/{anomaly_id}"
    
    payload = {
        "is_reviewed": is_reviewed
    }
    
    response = requests.patch(url, json=payload)
    return response.json()

def get_model_info(user_id):
    """Get information about a user's model"""
    url = f"{BASE_URL}/model-info/{user_id}"
    
    response = requests.get(url)
    return response.json()

def generate_sample_data(num_records=20, anomalies=False):
    """Generate sample data for training or prediction"""
    data = []
    
    for i in range(num_records):
        # Generate normal data
        if not anomalies:
            record = {
                "temperature": round(random.uniform(36.0, 37.5), 1),
                "speed": round(random.uniform(3.0, 7.0), 1),
                "heart_beat": random.randint(60, 100),
                "timestamp": datetime.datetime.now().isoformat()
            }
        # Generate anomalous data
        else:
            # 50% chance of high temperature
            if random.random() > 0.5:
                record = {
                    "temperature": round(random.uniform(38.0, 40.0), 1),
                    "speed": round(random.uniform(3.0, 7.0), 1),
                    "heart_beat": random.randint(60, 100),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            # 50% chance of high heart rate
            else:
                record = {
                    "temperature": round(random.uniform(36.0, 37.5), 1),
                    "speed": round(random.uniform(3.0, 7.0), 1),
                    "heart_beat": random.randint(120, 180),
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        data.append(record)
    
    return data

def pretty_print(json_data):
    """Pretty print JSON data"""
    print(json.dumps(json_data, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Health Monitoring API Client")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train a model for a user")
    train_parser.add_argument("--user", required=True, help="User ID")
    train_parser.add_argument("--local", action="store_true", help="Use generated sample data instead of Firebase")
    train_parser.add_argument("--samples", type=int, default=20, help="Number of samples to generate if using local data")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions for user data")
    predict_parser.add_argument("--user", required=True, help="User ID")
    predict_parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    predict_parser.add_argument("--anomalies", action="store_true", help="Generate anomalous data")
    
    # Get anomalies command
    anomalies_parser = subparsers.add_parser("anomalies", help="Get anomalies for a user")
    anomalies_parser.add_argument("--user", required=True, help="User ID")
    anomalies_parser.add_argument("--reviewed", choices=["true", "false"], help="Filter by review status")
    anomalies_parser.add_argument("--limit", type=int, default=100, help="Maximum number of anomalies to return")
    anomalies_parser.add_argument("--offset", type=int, default=0, help="Number of anomalies to skip")
    
    # Get anomaly details command
    details_parser = subparsers.add_parser("details", help="Get details for a specific anomaly")
    details_parser.add_argument("--user", required=True, help="User ID")
    details_parser.add_argument("--id", required=True, help="Anomaly ID")
    
    # Update anomaly status command
    update_parser = subparsers.add_parser("update", help="Update anomaly review status")
    update_parser.add_argument("--user", required=True, help="User ID")
    update_parser.add_argument("--id", required=True, help="Anomaly ID")
    update_parser.add_argument("--reviewed", choices=["true", "false"], required=True, help="New review status")
    
    # Get model info command
    info_parser = subparsers.add_parser("info", help="Get information about a user's model")
    info_parser.add_argument("--user", required=True, help="User ID")
    
    args = parser.parse_args()
    
    if args.command == "train":
        if args.local:
            sample_data = generate_sample_data(args.samples)
            result = train_model(args.user, use_firebase=False, sample_data=sample_data)
        else:
            result = train_model(args.user)
        pretty_print(result)
    
    elif args.command == "predict":
        sample_data = generate_sample_data(args.samples, args.anomalies)
        result = make_predictions(args.user, sample_data)
        pretty_print(result)
    
    elif args.command == "anomalies":
        reviewed = None
        if args.reviewed:
            reviewed = args.reviewed.lower() == "true"
        result = get_anomalies(args.user, reviewed, args.limit, args.offset)
        pretty_print(result)
    
    elif args.command == "details":
        result = get_anomaly_details(args.user, args.id)
        pretty_print(result)
    
    elif args.command == "update":
        is_reviewed = args.reviewed.lower() == "true"
        result = update_anomaly_status(args.user, args.id, is_reviewed)
        pretty_print(result)
    
    elif args.command == "info":
        result = get_model_info(args.user)
        pretty_print(result)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()