import firebase_admin
from firebase_admin import credentials, firestore
import json
import uuid

# Load Firebase SDK
cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load JSON data
with open("user123_data.json", "r") as f:
    data = json.load(f)

user_id = "user123"
collection_ref = db.collection("users").document(user_id).collection("data")

for record in data:
    doc_id = str(uuid.uuid4())  # Generate a unique ID for each record
    collection_ref.document(doc_id).set(record)

print(f"✅ Uploaded {len(data)} records to Firestore for user: {user_id}")
