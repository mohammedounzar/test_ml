import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase Admin
cred = credentials.Certificate('firebase-adminsdk.json')
firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# Read JSON file
with open('user123_data.json', 'r') as file:
    json_data = json.load(file)

# Get user_id and data
user_id = json_data['user_id']
data_points = json_data['data']

# Reference to user document
user_ref = db.collection('users').document(user_id)
user_ref.set({
    'user_id': user_id,
    'created_at': firestore.SERVER_TIMESTAMP
})

# Add data points to subcollection
data_collection = user_ref.collection('data')
for point in data_points:
    # Convert timestamp string to datetime
    timestamp = datetime.strptime(point['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
    
    # Add document to data subcollection
    data_collection.add({
        'temperature': point['temperature'],
        'heart_beat': point['heart_beat'],
        'speed': point['speed'],
        'timestamp': timestamp
    })

print(f"Successfully imported data for user {user_id}")